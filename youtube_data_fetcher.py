import os
from dotenv import load_dotenv
import pandas as pd
from googleapiclient.discovery import build
import json
from datetime import datetime
import time
import isodate
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
API_KEY = os.getenv('YOUTUBE_API_KEY')
if not API_KEY:
    raise ValueError("YouTube API key not found in environment variables")

QUOTA_COST = {
    'search': 100,
    'videos': 1
}
DAILY_QUOTA_LIMIT = 10000
VIDEOS_PER_KEYWORD = 50
DATA_DIR = 'youtube_data'
STATE_FILE = 'fetch_state.json'

class YouTubeDataFetcher:
    def __init__(self):
        try:
            self.youtube = build('youtube', 'v3', developerKey=API_KEY)
            self.quota_used = 0
            self.ensure_directories()
            self.state = self.load_state()
            logging.info("YouTubeDataFetcher initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing YouTubeDataFetcher: {str(e)}")
            raise

    def can_make_request(self, cost):
        """Check if we can make another API request"""
        return (self.quota_used + cost) <= DAILY_QUOTA_LIMIT

    def ensure_directories(self):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            logging.info(f"Created/verified directory: {DATA_DIR}")
        except Exception as e:
            logging.error(f"Error creating directory: {str(e)}")
            raise

    def load_state(self):
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                logging.info("State file loaded successfully")
                return state
            logging.info("No existing state file found, creating new state")
            return {
                'last_update': None,
                'processed_keywords': {}
            }
        except Exception as e:
            logging.error(f"Error loading state: {str(e)}")
            raise

    def save_state(self):
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
            logging.info("State saved successfully")
        except Exception as e:
            logging.error(f"Error saving state: {str(e)}")
            raise

    def fetch_videos_for_keyword(self, keyword, category):
        if not self.can_make_request(QUOTA_COST['search']):
            logging.warning("Daily quota limit reached")
            return None
        
        try:
            logging.info(f"Searching videos for keyword: {keyword}")
            search_response = self.youtube.search().list(
                q=keyword,
                part='id',
                type='video',
                maxResults=VIDEOS_PER_KEYWORD,
                order='relevance'
            ).execute()
            
            self.quota_used += QUOTA_COST['search']
            
            video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            
            if not video_ids:
                logging.warning(f"No videos found for keyword: {keyword}")
                return None
            
            if not self.can_make_request(len(video_ids) * QUOTA_COST['videos']):
                logging.warning("Quota limit would be exceeded fetching video details")
                return None
                
            videos_response = self.youtube.videos().list(
                id=','.join(video_ids),
                part='snippet,statistics,contentDetails'
            ).execute()
            
            self.quota_used += len(video_ids) * QUOTA_COST['videos']
            
            videos_data = []
            for video in videos_response.get('items', []):
                try:
                    duration = isodate.parse_duration(video['contentDetails']['duration']).total_seconds()
                    
                    video_data = {
                        'keyword': keyword,
                        'category': category,
                        'video_id': video['id'],
                        'title': video['snippet']['title'],
                        'published_date': video['snippet']['publishedAt'],
                        'duration_seconds': duration,
                        'view_count': int(video['statistics'].get('viewCount', 0)),
                        'like_count': int(video['statistics'].get('likeCount', 0)),
                        'comment_count': int(video['statistics'].get('commentCount', 0))
                    }
                    videos_data.append(video_data)
                except Exception as e:
                    logging.error(f"Error processing video {video.get('id', 'unknown')}: {str(e)}")
                    continue
            
            logging.info(f"Successfully fetched {len(videos_data)} videos for keyword: {keyword}")
            return videos_data
            
        except Exception as e:
            logging.error(f"Error fetching videos for keyword '{keyword}': {str(e)}")
            return None

    def process_keywords(self):
        try:
            logging.info("Starting to process keywords")
            keywords_df = pd.read_csv('keywords.csv')
            logging.info(f"Loaded {len(keywords_df)} keywords from CSV")
            
            main_data_file = os.path.join(DATA_DIR, 'all_videos_data.csv')
            if os.path.exists(main_data_file):
                all_data = pd.read_csv(main_data_file)
                logging.info(f"Loaded existing data with {len(all_data)} records")
            else:
                all_data = pd.DataFrame()
                logging.info("Created new DataFrame for data")
            
            for _, row in keywords_df.iterrows():
                keyword = row['keyword']
                category = row['group']
                
                if keyword in self.state['processed_keywords']:
                    logging.info(f"Skipping already processed keyword: {keyword}")
                    continue
                
                if not self.can_make_request(QUOTA_COST['search']):
                    logging.warning("Daily quota limit reached. Stopping for today.")
                    break
                
                logging.info(f"Processing: {keyword} ({category})")
                videos_data = self.fetch_videos_for_keyword(keyword, category)
                
                if videos_data:
                    new_data = pd.DataFrame(videos_data)
                    all_data = pd.concat([all_data, new_data], ignore_index=True)
                    all_data.to_csv(main_data_file, index=False)
                    
                    self.state['processed_keywords'][keyword] = {
                        'processed_date': datetime.now().isoformat(),
                        'videos_count': len(videos_data)
                    }
                    self.save_state()
                    
                time.sleep(1)
            
            logging.info(f"Processing complete. Quota used: {self.quota_used}")
            return all_data
            
        except Exception as e:
            logging.error(f"Error in process_keywords: {str(e)}")
            raise

def main():
    try:
        logging.info("Starting YouTube data fetching process")
        fetcher = YouTubeDataFetcher()
        data = fetcher.process_keywords()
        logging.info("Data collection complete!")
        logging.info(f"Total videos collected: {len(data) if data is not None else 0}")
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 