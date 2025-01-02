# YouTube Data Analysis Project

## Problem Statement
Understanding the trends and impact of plant breeding research over time using YouTube as a platform to analyze historical, current, and modern topics. The project aims to utilize the YouTube API to gather data and leverage machine learning techniques to predict future research directions.

## Objective
- Collect data from YouTube using the YouTube API based on specific plant breeding keywords.
- Analyze the collected data to determine trends in viewership, engagement, and sentiment.
- Identify emerging topics and predict future directions in plant breeding research using machine learning models.
- Present the insights visually for better interpretability.

## Technologies Used
- **API and Data Handling:**
  - YouTube Data API v3
  - Python libraries: `pandas`, `numpy`, `datetime`
- **Visualization and Analysis:**
  - `matplotlib`, `seaborn`, `plotly`
- **Machine Learning and NLP:**
  - `scikit-learn`, `NLTK`, `VADER Sentiment Analysis`
- **Data Storage and Preprocessing:**
  - CSV for structured data storage
  - `SQLite` for query-based data management


## Discussion on Results
1. **Data Cleaning and Preprocessing:**
   - Removed duplicates and standardized column formats.
   - Created new features like human-readable durations and engagement rates.
2. **Trend Analysis:**
   - Identified increasing interest in modern plant breeding techniques like CRISPR and AI-based genomic selection.
   - Observed consistent engagement with traditional topics, suggesting continued relevance.
3. **Sentiment Analysis:**
   - Sentiment scores indicated positive perceptions of cutting-edge research but mixed opinions about traditional methods.
4. **Keyword Analysis:**
   - High-frequency keywords in descriptions and tags showed significant overlap between current and modern methods.
5. **Predictive Modeling:**
   - Models trained on historical data predicted increasing focus on climate-resilient and precision breeding topics.

## Key Findings
- **Historical Insights:** Traditional methods like mass selection and pedigree breeding remain foundational.
- **Current Trends:** Conventional methods such as marker-assisted selection and QTL mapping show high engagement.
- **Emerging Topics:** Precision agriculture, CRISPR, and machine learning dominate modern research discussions.
- **Engagement Metrics:** Videos focusing on cutting-edge methods receive higher engagement rates than traditional topics.

## Relevance of Such Analysis
- **Educational Value:** Helps researchers and students understand the evolution of plant breeding methodologies.
- **Decision-Making Support:** Guides funding agencies and institutions to allocate resources towards emerging technologies.
- **Public Awareness:** Highlights the growing influence of technology-driven agriculture.
- **Future Research:** Enables early identification of trends and gaps in plant breeding research.

## Future Work
- Expand the dataset to include more diverse keywords and languages.
- Implement advanced NLP models for topic modeling and sentiment analysis.
- Explore video content (transcripts) for deeper insights.

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Project Setup
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Environment Setup
1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```   

### 3. API Configuration
1. Get a YouTube Data API key from the [Google Cloud Console](https://console.cloud.google.com/)
2. Set up environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your YouTube API key:
     ```
     YOUTUBE_API_KEY=your_actual_api_key_here
     ```

### 4. Run the Application
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python dashboard.py
   ```

## Project Structure
├── README.md
├── requirements.txt
├── .env.example
├── .env # (git-ignored)
├── .gitignore
├── youtube_data_fetcher.py
├── dashboard.py
└── youtube_data/ # (git-ignored)
└── all_videos_data.csv

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Security Notice
⚠️ If you're forking this repository, make sure to:
1. Never commit API keys or sensitive data
2. Use environment variables for sensitive data
3. Check git history for sensitive data before making public
