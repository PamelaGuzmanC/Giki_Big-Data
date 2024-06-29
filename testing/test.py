import feedparser as fp
import dateutil.parser
import newspaper
from newspaper import Article
import logging
import pandas as pd
import json
from datetime import datetime, timedelta

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Helper:
    @staticmethod
    def print_scrape_status(count):
        logging.info(f'Scraped {count} articles so far...')

class Scraper:
    def __init__(self, sources, start_date, end_date):
        self.sources = sources
        self.start_date = start_date
        self.end_date = end_date

    def scrape(self):
        # Function that scrapes the content from the URLs in the source data
        try:
            articles_list = []
            current_date = self.start_date

            while current_date <= self.end_date:
                for source, content in self.sources.items():
                    for url in content['rss']:
                        logging.info(f'Processing RSS feed: {url}')
                        d = fp.parse(url)
                        for entry in d.entries:
                            if hasattr(entry, 'published'):
                                article_date = dateutil.parser.parse(getattr(entry, 'published'))
                                logging.info(f'Found article with date: {article_date}')
                                if article_date.strftime('%Y-%m-%d') == current_date.strftime('%Y-%m-%d'):
                                    try:
                                        logging.info(f'Processing article: {entry.link}')
                                        content = Article(entry.link)
                                        content.download()
                                        content.parse()
                                        content.nlp()
                                        try:
                                            article = {
                                                'source': source,
                                                'url': entry.link,
                                                'date': article_date.strftime('%Y-%m-%d'),
                                                'time': article_date.strftime('%H:%M:%S %Z'),  # hour, minute, timezone (converted)
                                                'title': content.title,
                                                'body': content.text,
                                                'summary': content.summary,
                                                'keywords': content.keywords,
                                                'image_url': content.top_image
                                            }
                                            articles_list.append(article)
                                            Helper.print_scrape_status(len(articles_list))
                                        except Exception as e:
                                            logging.error(f'Error processing article: {e}')
                                            logging.info('Continuing...')
                                    except Exception as e:
                                        logging.error(f'Error downloading/parsing article: {e}')
                                        logging.info('Continuing...')
                current_date += timedelta(days=1)

            # Check if any articles were scraped
            if articles_list:
                logging.info(f'Total articles scraped: {len(articles_list)}')
                
                # Convert articles list to DataFrame and save to CSV
                df = pd.DataFrame(articles_list)
                df.to_csv('scraped_articles.csv', index=False)
                logging.info('Scraped articles saved to scraped_articles.csv')
            else:
                logging.warning('No articles were scraped. Check the sources and date provided.')

            return articles_list
        except Exception as e:
            logging.error(f'Error in "Scraper.scrape()": {e}')
            raise Exception(f'Error in "Scraper.scrape()": {e}')

# Load sources from the JSON file
def load_sources(json_file):
    try:
        with open(json_file, 'r') as f:
            sources = json.load(f)
        return sources
    except Exception as e:
        logging.error(f'Error loading sources from {json_file}: {e}')
        raise

# Example usage:
if __name__ == '__main__':
    sources_file = 'sources.json'
    start_date = datetime.strptime('2024-06-20', '%Y-%m-%d')
    end_date = datetime.strptime('2024-06-23', '%Y-%m-%d')

    sources = load_sources(sources_file)
    scraper = Scraper(sources, start_date, end_date)
    scraper.scrape()
