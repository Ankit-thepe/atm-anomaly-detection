# freeform_labeler_5videos.py
import cv2
import pandas as pd
import os
from collections import Counter

class ATMVideoLabeler:
    def __init__(self, videos_dir, clip_duration=4):
        self.videos_dir = videos_dir
        self.clip_duration = clip_duration
        self.labels_df = pd.DataFrame(columns=['video_path', 'clip_start', 'clip_end', 'activity', 'clip_id'])
        self.activity_classes = set()
        
    def get_video_list(self):
        """Get list of video files in the directory"""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.MP4')
        video_files = [f for f in os.listdir(self.videos_dir) 
                      if f.endswith(video_extensions)]
        return sorted(video_files)
    
    def get_video_info(self, video_path):
        """Get video information"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        return fps, total_frames, duration
    
    def select_videos_to_label(self):
        """Let user select which videos to label"""
        video_files = self.get_video_list()
        
        print("🎥 Found videos in directory:")
        for i, video_file in enumerate(video_files):
            video_path = os.path.join(self.videos_dir, video_file)
            fps, total_frames, duration = self.get_video_info(video_path)
            if fps:
                print(f"{i+1}. {video_file} ({duration:.1f}s, {total_frames} frames)")
            else:
                print(f"{i+1}. {video_file} (Error reading video)")
        
        print("\nSelect videos to label (e.g., '1 3 5' or 'all'):")
        selection = input().strip()
        
        if selection.lower() == 'all':
            return video_files
        else:
            try:
                indices = [int(x)-1 for x in selection.split()]
                return [video_files[i] for i in indices if i < len(video_files)]
            except:
                print("Invalid selection. Labeling all videos.")
                return video_files
    
    def play_and_label_clip(self, cap, start_frame, end_frame, video_file, clip_id, total_clips):
        """Play clip and get activity label from user"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(f"\n{'='*60}")
        print(f"🎬 Video: {video_file} | Clip {clip_id+1}/{total_clips}")
        print(f"📏 Frames: {start_frame}-{end_frame}")
        print("Playing clip... Press 'q' to stop and label")
        print("="*60)
        
        # Play the clip
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add informative overlay
            cv2.putText(frame, f"Video: {video_file}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Clip: {clip_id+1}/{total_clips}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {current_frame}/{end_frame}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to stop & label", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('ATM Activity Labeling - Press Q to Label', frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            current_frame += 1
        
        cv2.destroyAllWindows()
        
        # Get activity label
        print("\n📝 Describe the activity in this clip:")
        print("Examples: 'person_entering', 'withdrawing_cash', 'suspicious_loitering'")
        print("Commands: 'skip', 'back', 'quit', 'list'")
        
        while True:
            activity = input("Activity description: ").strip()
            
            if activity.lower() == 'list':
                if self.activity_classes:
                    print("Activities used so far:", sorted(self.activity_classes))
                else:
                    print("No activities labeled yet.")
                continue
            elif activity.lower() in ['skip', 'back', 'quit']:
                return activity.lower()
            elif activity:
                return activity
            else:
                print("Please enter an activity description!")
    
    def label_video(self, video_file):
        """Label all clips in a video"""
        video_path = os.path.join(self.videos_dir, video_file)
        
        fps, total_frames, duration = self.get_video_info(video_path)
        if not fps:
            print(f"❌ Error reading video: {video_file}")
            return
        
        frames_per_clip = int(fps * self.clip_duration)
        total_clips = (total_frames + frames_per_clip - 1) // frames_per_clip
        
        print(f"\n📊 Video Info: {duration:.1f}s, {total_frames} frames, {total_clips} clips")
        
        cap = cv2.VideoCapture(video_path)
        clip_id = 0
        clip_history = []  # Track recent clips for 'back' functionality
        
        while clip_id < total_clips:
            start_frame = clip_id * frames_per_clip
            end_frame = min((clip_id + 1) * frames_per_clip, total_frames)
            
            activity = self.play_and_label_clip(cap, start_frame, end_frame, 
                                              video_file, clip_id, total_clips)
            
            if activity == 'quit':
                break
            elif activity == 'back':
                if clip_history:
                    # Remove last labeled clip
                    last_clip = clip_history.pop()
                    self.labels_df = self.labels_df[self.labels_df['clip_id'] != last_clip['clip_id']]
                    clip_id = last_clip['clip_number']
                    print(f"↩️  Went back to clip {clip_id}")
                else:
                    print("No previous clip to go back to!")
                continue
            elif activity == 'skip':
                print("⏭️  Skipped clip")
                clip_id += 1
                continue
            
            # Save the labeled clip
            clip_data = {
                'video_path': video_path,
                'clip_start': start_frame,
                'clip_end': end_frame,
                'activity': activity,
                'clip_id': f"{video_file}_clip{clip_id:04d}"
            }
            
            self.labels_df = pd.concat([self.labels_df, pd.DataFrame([clip_data])], ignore_index=True)
            self.activity_classes.add(activity)
            clip_history.append({'clip_number': clip_id, 'clip_id': clip_data['clip_id']})
            
            print(f"✅ Labeled as: {activity}")
            clip_id += 1
        
        cap.release()
        cv2.destroyAllWindows()
    
    def show_progress(self):
        """Show labeling progress"""
        if self.labels_df.empty:
            print("No labels yet!")
            return
        
        print(f"\n📈 Labeling Progress:")
        print(f"Total clips labeled: {len(self.labels_df)}")
        print(f"Unique activities: {len(self.activity_classes)}")
        
        # Show activity distribution
        activity_counts = self.labels_df['activity'].value_counts()
        print("\nActivity Distribution:")
        for activity, count in activity_counts.head(10).items():  # Show top 10
            percentage = (count / len(self.labels_df)) * 100
            print(f"  {activity}: {count} clips ({percentage:.1f}%)")
        
        if len(activity_counts) > 10:
            print(f"  ... and {len(activity_counts) - 10} more activities")
    
    def save_labels(self, output_path):
        """Save labels to CSV"""
        if self.labels_df.empty:
            print("No labels to save!")
            return
        
        self.labels_df.to_csv(output_path, index=False)
        print(f"💾 Labels saved to: {output_path}")
        
        # Save activity classes
        classes_path = output_path.replace('.csv', '_classes.txt')
        with open(classes_path, 'w') as f:
            for i, activity in enumerate(sorted(self.activity_classes)):
                f.write(f"{i}: {activity}\n")
        print(f"💾 Activity classes saved to: {classes_path}")
        
        self.show_progress()

def main():
    print("🎯 ATM Video Labeling Tool - 5 Video Experiment")
    print("="*60)
    
    # Initialize labeler
    labeler = ATMVideoLabeler("/home/mitsdu/atm_experiment/videos", clip_duration=4)
    
    # Select videos to label
    videos_to_label = labeler.select_videos_to_label()
    print(f"\nSelected {len(videos_to_label)} videos for labeling")
    
    # Label each video
    for i, video_file in enumerate(videos_to_label):
        print(f"\n🎥 Processing video {i+1}/{len(videos_to_label)}: {video_file}")
        
        labeler.label_video(video_file)
        labeler.save_labels(f"/home/mitsdu/atm_experiment/annotations/progress_{i+1}.csv")
        
        if i < len(videos_to_label) - 1:
            cont = input("\nContinue to next video? (y/n): ").lower()
            if cont != 'y':
                break
    
    # Final save
    labeler.save_labels("/home/mitsdu/atm_experiment/annotations/final_labels_5videos.csv")
    print("\n🎉 Labeling completed! Ready for training.")

if __name__ == "__main__":
    main()
