# app/api/routes.py
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import uuid
import json
from datetime import datetime
import pandas as pd
from flask_jwt_extended import jwt_required, get_jwt_identity

from app import db
from app.models.dataset import Dataset
from app.modules.data_processor.preprocessor import DataPreprocessor
from app.modules.data_processor.feature_engineering import FeatureEngineer
from app.modules.visualization.dashboard_generator import DashboardGenerator
from app.modules.market_simulator.simulator import MarketSimulator
from app.modules.ai_copilot.strategy_advisor import StrategyAdvisor
from app.modules.ai_copilot.swot_analyzer import SWOTAnalyzer
from app.modules.trend_detector.scraper import TrendScraper
from app.modules.trend_detector.trend_analyzer import TrendAnalyzer
from app.modules.sustainability.ethical_analyzer import EthicalAnalyzer
from app.modules.sustainability.sustainability_metrics import SustainabilityMetrics
from app.modules.risk_manager.risk_predictor import RiskPredictor
from app.modules.risk_manager.mitigation_strategies import MitigationStrategies
from app.modules.recommender.business_recommender import BusinessRecommender
from app.modules.recommender.growth_strategies import GrowthStrategies

api_bp = Blueprint('api', __name__)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

# Dataset Routes
@api_bp.route('/datasets', methods=['POST'])
@jwt_required()
def upload_dataset():
    """Upload a new dataset"""
    current_user_id = get_jwt_identity()
    
    # Check if file part exists
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Check if file type is allowed
    if file and allowed_file(file.filename):
        # Generate secure filename and unique path
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        
        # Save file
        file.save(file_path)
        
        # Get file metadata
        file_size = os.path.getsize(file_path)
        file_type = filename.rsplit('.', 1)[1].lower()
        
        # Create dataset record
        dataset = Dataset(
            id=file_id,
            name=request.form.get('name', filename),
            description=request.form.get('description', ''),
            file_path=file_path,
            file_type=file_type,
            size_bytes=file_size,
            columns="{}",  # Empty JSON object
            row_count=0,
            is_processed=False,
            processing_status='pending',
            user_id=current_user_id
        )
        
        # Load dataset to extract metadata
        try:
            preprocessor = DataPreprocessor()
            df = preprocessor.load_dataset(file_path)
            
            # Update dataset with metadata
            dataset.row_count = len(df)
            dataset.set_columns({col: str(df[col].dtype) for col in df.columns})
            
            # Auto-preprocess if requested
            if request.form.get('auto_preprocess', 'false').lower() == 'true':
                preprocessor.auto_preprocess()
                dataset.is_processed = True
                dataset.processing_status = 'completed'
                dataset.set_preprocessing_steps(preprocessor.preprocessing_steps)
            
            db.session.add(dataset)
            db.session.commit()
            
            return jsonify({
                'message': 'Dataset uploaded successfully',
                'dataset': dataset.to_dict()
            }), 201
            
        except Exception as e:
            return jsonify({
                'error': f'Error processing dataset: {str(e)}'
            }), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@api_bp.route('/datasets', methods=['GET'])
@jwt_required()
def get_datasets():
    """Get all datasets for the current user"""
    current_user_id = get_jwt_identity()
    
    datasets = Dataset.query.filter_by(user_id=current_user_id).all()
    return jsonify({
        'datasets': [dataset.to_dict() for dataset in datasets]
    }), 200

@api_bp.route('/datasets/<dataset_id>', methods=['GET'])
@jwt_required()
def get_dataset(dataset_id):
    """Get a specific dataset"""
    current_user_id = get_jwt_identity()
    
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user_id).first()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
        
    return jsonify({
        'dataset': dataset.to_dict()
    }), 200

@api_bp.route('/datasets/<dataset_id>/preview', methods=['GET'])
@jwt_required()
def preview_dataset(dataset_id):
    """Preview dataset contents"""
    current_user_id = get_jwt_identity()
    
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user_id).first()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        rows = int(request.args.get('rows', 10))
        
        # Load dataset
        df = pd.read_csv(dataset.file_path) if dataset.file_type == 'csv' else \
             pd.read_excel(dataset.file_path) if dataset.file_type in ['xlsx', 'xls'] else \
             pd.read_json(dataset.file_path) if dataset.file_type == 'json' else \
             pd.read_parquet(dataset.file_path)
        
        # Get preview data
        preview_data = df.head(rows).to_dict(orient='records')
        
        # Get column statistics
        stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {
                    'type': 'numeric',
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'missing': int(df[col].isna().sum())
                }
            else:
                stats[col] = {
                    'type': 'categorical',
                    'unique_values': int(df[col].nunique()),
                    'top_value': str(df[col].value_counts().index[0]) if not df[col].value_counts().empty else None,
                    'missing': int(df[col].isna().sum())
                }
        
        return jsonify({
            'preview': preview_data,
            'statistics': stats,
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error previewing dataset: {str(e)}'
        }), 500

@api_bp.route('/datasets/<dataset_id>/process', methods=['POST'])
@jwt_required()
def process_dataset(dataset_id):
    """Process a dataset with specified preprocessing steps"""
    current_user_id = get_jwt_identity()
    
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user_id).first()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    # Get preprocessing steps from request
    preprocessing_steps = request.json.get('preprocessing_steps', [])
    
    # Check if async processing is requested
    async_processing = request.json.get('async', False)
    
    if async_processing:
        # Start asynchronous processing
        from app.tasks.data_processing import process_dataset_async
        task = process_dataset_async.delay(dataset_id, preprocessing_steps)
        
        # Update dataset status
        dataset.processing_status = 'queued'
        db.session.commit()
        
        return jsonify({
            'message': 'Dataset processing queued',
            'task_id': task.id,
            'dataset': dataset.to_dict()
        }), 202
    else:
        # Synchronous processing (as before)
        try:
            # Load dataset
            preprocessor = DataPreprocessor()
            df = preprocessor.load_dataset(dataset.file_path)
            
            # Apply each preprocessing step
            for step in preprocessing_steps:
                step_type = step.get('type')
                params = step.get('params', {})
                
                if step_type == 'handle_missing_values':
                    preprocessor.handle_missing_values(params)
                elif step_type == 'handle_outliers':
                    preprocessor.handle_outliers(params.get('method'), params.get('threshold'))
                elif step_type == 'encode_categorical_features':
                    preprocessor.encode_categorical_features(params.get('method'), params.get('max_categories'))
                elif step_type == 'normalize_numerical_features':
                    preprocessor.normalize_numerical_features(params.get('method'))
                elif step_type == 'process_datetime_features':
                    preprocessor.process_datetime_features()
            
            # Update dataset with preprocessing steps
            dataset.is_processed = True
            dataset.processing_status = 'completed'
            dataset.set_preprocessing_steps(preprocessor.preprocessing_steps)
            
            db.session.commit()
            
            return jsonify({
                'message': 'Dataset processed successfully',
                'dataset': dataset.to_dict(),
                'preprocessing_summary': preprocessor.get_data_summary()
            }), 200
            
        except Exception as e:
            # Update dataset with error status
            dataset.processing_status = 'error'
            db.session.commit()
            
            return jsonify({
                'error': f'Error processing dataset: {str(e)}'
            }), 500
        
@api_bp.route('/datasets/<dataset_id>/dashboard', methods=['POST'])
@jwt_required()
def generate_dashboard(dataset_id):
    """Generate a dashboard for a dataset"""
    current_user_id = get_jwt_identity()
    
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user_id).first()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        # Load dataset
        df = pd.read_csv(dataset.file_path) if dataset.file_type == 'csv' else \
             pd.read_excel(dataset.file_path) if dataset.file_type in ['xlsx', 'xls'] else \
             pd.read_json(dataset.file_path) if dataset.file_type == 'json' else \
             pd.read_parquet(dataset.file_path)
        
        # Initialize dashboard generator
        dashboard_generator = DashboardGenerator(df)
        
        # Generate dashboard based on request type
        dashboard_type = request.json.get('type', 'auto')
        
        if dashboard_type == 'auto':
            dashboard = dashboard_generator.auto_generate_dashboard()
        elif dashboard_type == 'custom':
            prompt = request.json.get('prompt', '')
            dashboard = dashboard_generator.generate_dashboard_from_prompt(prompt)
        else:
            return jsonify({'error': 'Invalid dashboard type'}), 400
        
        return jsonify({
            'dashboard': dashboard
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating dashboard: {str(e)}'
        }), 500

@api_bp.route('/datasets/<dataset_id>/charts', methods=['POST'])
@jwt_required()
def create_chart(dataset_id):
    """Create a specific chart from dataset"""
    current_user_id = get_jwt_identity()
    
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user_id).first()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    chart_type = request.json.get('chart_type')
    params = request.json.get('params', {})
    
    if not chart_type:
        return jsonify({'error': 'Chart type is required'}), 400
    
    try:
        # Load dataset
        df = pd.read_csv(dataset.file_path) if dataset.file_type == 'csv' else \
             pd.read_excel(dataset.file_path) if dataset.file_type in ['xlsx', 'xls'] else \
             pd.read_json(dataset.file_path) if dataset.file_type == 'json' else \
             pd.read_parquet(dataset.file_path)
        
        # Initialize dashboard generator
        dashboard_generator = DashboardGenerator(df)
        
        # Create chart based on type
        chart = None
        
        if chart_type == 'histogram':
            chart = dashboard_generator.create_histogram(
                params.get('column'),
                params.get('bins', 30),
                params.get('title')
            )
        elif chart_type == 'bar_chart':
            chart = dashboard_generator.create_bar_chart(
                params.get('column'),
                params.get('top_n', 10),
                params.get('title'),
                params.get('horizontal', False)
            )
        elif chart_type == 'pie_chart':
            chart = dashboard_generator.create_pie_chart(
                params.get('column'),
                params.get('top_n', 7),
                params.get('title')
            )
        elif chart_type == 'scatter_plot':
            chart = dashboard_generator.create_scatter_plot(
                params.get('x_column'),
                params.get('y_column'),
                params.get('color_column'),
                params.get('size_column'),
                params.get('title')
            )
        elif chart_type == 'line_chart':
            chart = dashboard_generator.create_line_chart(
                params.get('x_column'),
                params.get('y_columns', []),
                params.get('title')
            )
        elif chart_type == 'box_plot':
            chart = dashboard_generator.create_box_plot(
                params.get('x_column'),
                params.get('y_column'),
                params.get('title')
            )
        elif chart_type == 'heatmap':
            chart = dashboard_generator.create_heatmap(
                params.get('columns', []),
                params.get('title')
            )
        else:
            return jsonify({'error': 'Unsupported chart type'}), 400
        
        return jsonify({
            'chart': chart
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error creating chart: {str(e)}'
        }), 500

# Market Simulator Routes
@api_bp.route('/market-simulator/price-change', methods=['POST'])
@jwt_required()
def simulate_price_change():
    """Simulate price change impact"""
    try:
        # Get simulation parameters
        company_data = request.json.get('company_data', {})
        market_data = request.json.get('market_data', {})
        price_change_pct = request.json.get('price_change_pct', 0)
        periods = request.json.get('periods', 12)
        
        # Initialize simulator
        simulator = MarketSimulator()
        simulator.load_company_data(company_data)
        simulator.load_market_data(market_data)
        
        # Run simulation
        result = simulator.simulate_price_change(price_change_pct, periods)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error simulating price change: {str(e)}'
        }), 500

@api_bp.route('/market-simulator/marketing-spend', methods=['POST'])
@jwt_required()
def simulate_marketing_spend():
    """Simulate marketing spend impact"""
    try:
        # Get simulation parameters
        company_data = request.json.get('company_data', {})
        market_data = request.json.get('market_data', {})
        spend_change_pct = request.json.get('spend_change_pct', 0)
        periods = request.json.get('periods', 12)
        
        # Initialize simulator
        simulator = MarketSimulator()
        simulator.load_company_data(company_data)
        simulator.load_market_data(market_data)
        
        # Run simulation
        result = simulator.simulate_marketing_spend_change(spend_change_pct, periods)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error simulating marketing spend change: {str(e)}'
        }), 500

@api_bp.route('/market-simulator/product-launch', methods=['POST'])
@jwt_required()
def simulate_product_launch():
    """Simulate product launch impact"""
    try:
        # Get simulation parameters
        company_data = request.json.get('company_data', {})
        market_data = request.json.get('market_data', {})
        product_data = request.json.get('product_data', {})
        periods = request.json.get('periods', 24)
        
        # Initialize simulator
        simulator = MarketSimulator()
        simulator.load_company_data(company_data)
        simulator.load_market_data(market_data)
        
        # Run simulation
        result = simulator.simulate_product_launch(product_data, periods)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error simulating product launch: {str(e)}'
        }), 500

@api_bp.route('/market-simulator/monte-carlo', methods=['POST'])
@jwt_required()
def run_monte_carlo_simulation():
    """Run Monte Carlo simulation"""
    try:
        # Get simulation parameters
        simulation_type = request.json.get('simulation_type', 'price_change')
        parameters = request.json.get('parameters', {})
        iterations = request.json.get('iterations', 100)
        company_data = request.json.get('company_data', {})
        market_data = request.json.get('market_data', {})
        
        # Initialize simulator
        simulator = MarketSimulator()
        simulator.load_company_data(company_data)
        simulator.load_market_data(market_data)
        
        # Run simulation
        result = simulator.generate_monte_carlo_simulation(simulation_type, parameters, iterations)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error running Monte Carlo simulation: {str(e)}'
        }), 500

# AI Co-Pilot Routes
@api_bp.route('/ai-copilot/swot-analysis', methods=['POST'])
@jwt_required()
def generate_swot_analysis():
    """Generate SWOT analysis"""
    try:
        # Get business data
        business_data = request.json.get('business_data', {})
        industry_data = request.json.get('industry_data', {})
        competitor_data = request.json.get('competitor_data', {})
        
        # Initialize SWOT analyzer
        api_key = current_app.config.get('OPENAI_API_KEY')
        analyzer = SWOTAnalyzer(api_key)
        
        # Generate SWOT analysis
        result = analyzer.analyze_company_data(business_data, industry_data, competitor_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating SWOT analysis: {str(e)}'
        }), 500

@api_bp.route('/ai-copilot/strategic-recommendations', methods=['POST'])
@jwt_required()
def generate_strategic_recommendations():
    """Generate strategic recommendations"""
    try:
        # Get business data
        business_data = request.json.get('business_data', {})
        industry_data = request.json.get('industry_data', {})
        market_data = request.json.get('market_data', {})
        competitor_data = request.json.get('competitor_data', {})
        financial_data = request.json.get('financial_data', {})
        focus_area = request.json.get('focus_area')
        
        # Initialize strategy advisor
        api_key = current_app.config.get('OPENAI_API_KEY')
        advisor = StrategyAdvisor(api_key)
        
        # Set data
        advisor.set_company_data(business_data)
        if industry_data:
            advisor.set_industry_data(industry_data)
        if market_data:
            advisor.set_market_data(market_data)
        if competitor_data:
            advisor.set_competitor_data(competitor_data)
        if financial_data:
            advisor.set_financial_data(financial_data)
        
        # Generate recommendations
        result = advisor.generate_strategic_recommendations(focus_area)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating strategic recommendations: {str(e)}'
        }), 500

@api_bp.route('/ai-copilot/market-trends', methods=['POST'])
@jwt_required()
def get_market_trends():
    """Get market trends analysis"""
    try:
        # Get parameters
        industry = request.json.get('industry')
        
        # Initialize strategy advisor
        api_key = current_app.config.get('OPENAI_API_KEY')
        advisor = StrategyAdvisor(api_key)
        
        # Get trends
        result = advisor.get_market_trends(industry)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error getting market trends: {str(e)}'
        }), 500

@api_bp.route('/ai-copilot/business-query', methods=['POST'])
@jwt_required()
def answer_business_query():
    """Answer business query"""
    try:
        # Get query and context
        query = request.json.get('query')
        business_data = request.json.get('business_data', {})
        industry_data = request.json.get('industry_data', {})
        market_data = request.json.get('market_data', {})
        competitor_data = request.json.get('competitor_data', {})
        financial_data = request.json.get('financial_data', {})
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Initialize strategy advisor
        api_key = current_app.config.get('OPENAI_API_KEY')
        advisor = StrategyAdvisor(api_key)
        
        # Set data
        advisor.set_company_data(business_data)
        if industry_data:
            advisor.set_industry_data(industry_data)
        if market_data:
            advisor.set_market_data(market_data)
        if competitor_data:
            advisor.set_competitor_data(competitor_data)
        if financial_data:
            advisor.set_financial_data(financial_data)
        
        # Answer query
        result = advisor.answer_business_query(query)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error answering business query: {str(e)}'
        }), 500

@api_bp.route('/ai-copilot/business-plan', methods=['POST'])
@jwt_required()
def generate_business_plan():
    """Generate business plan"""
    try:
        # Get parameters
        plan_type = request.json.get('plan_type', 'strategic')
        timeframe = request.json.get('timeframe', '1-year')
        business_data = request.json.get('business_data', {})
        industry_data = request.json.get('industry_data', {})
        market_data = request.json.get('market_data', {})
        competitor_data = request.json.get('competitor_data', {})
        financial_data = request.json.get('financial_data', {})
        
        # Initialize strategy advisor
        api_key = current_app.config.get('OPENAI_API_KEY')
        advisor = StrategyAdvisor(api_key)
        
        # Set data
        advisor.set_company_data(business_data)
        if industry_data:
            advisor.set_industry_data(industry_data)
        if market_data:
            advisor.set_market_data(market_data)
        if competitor_data:
            advisor.set_competitor_data(competitor_data)
        if financial_data:
            advisor.set_financial_data(financial_data)
        
        # Generate business plan
        result = advisor.generate_business_plan(plan_type, timeframe)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating business plan: {str(e)}'
        }), 500

# Trend Detector Routes
@api_bp.route('/trend-detector/news', methods=['POST'])
@jwt_required()
def scrape_news():
    """Scrape news articles for trends"""
    try:
        # Get parameters
        keywords = request.json.get('keywords', [])
        days_back = request.json.get('days_back', 7)
        limit = request.json.get('limit', 100)
        
        if not keywords:
            return jsonify({'error': 'Keywords are required'}), 400
        
        # Initialize scraper
        news_api_key = current_app.config.get('NEWS_API_KEY')
        twitter_api_key = current_app.config.get('TWITTER_API_KEY')
        scraper = TrendScraper(news_api_key, twitter_api_key)
        
        # Scrape news
        articles = scraper.scrape_news_articles(keywords, days_back, limit)
        
        # Analyze trends
        analyzer = TrendAnalyzer()
        analysis = analyzer.analyze_news_articles(articles)
        
        return jsonify({
            'articles': articles,
            'analysis': analysis
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error scraping news: {str(e)}'
        }), 500

@api_bp.route('/trend-detector/social-media', methods=['POST'])
@jwt_required()
def scrape_social_media():
    """Scrape social media for trends"""
    try:
        # Get parameters
        keywords = request.json.get('keywords', [])
        days_back = request.json.get('days_back', 7)
        
        if not keywords:
            return jsonify({'error': 'Keywords are required'}), 400
        
        # Initialize scraper
        news_api_key = current_app.config.get('NEWS_API_KEY')
        twitter_api_key = current_app.config.get('TWITTER_API_KEY')
        scraper = TrendScraper(news_api_key, twitter_api_key)
        
        # Scrape social media
        social_data = scraper.scrape_social_media_trends(keywords, days_back)
        
        # Analyze trends
        analyzer = TrendAnalyzer()
        analysis = analyzer.analyze_social_trends(social_data)
        
        return jsonify({
            'social_data': social_data,
            'analysis': analysis
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error scraping social media: {str(e)}'
        }), 500

@api_bp.route('/trend-detector/industry-reports', methods=['POST'])
@jwt_required()
def scrape_industry_reports():
    """Scrape industry reports"""
    try:
        # Get parameters
        industry = request.json.get('industry')
        limit = request.json.get('limit', 5)
        
        if not industry:
            return jsonify({'error': 'Industry is required'}), 400
        
        # Initialize scraper
        news_api_key = current_app.config.get('NEWS_API_KEY')
        twitter_api_key = current_app.config.get('TWITTER_API_KEY')
        scraper = TrendScraper(news_api_key, twitter_api_key)
        
        # Scrape industry reports
        reports = scraper.scrape_industry_reports(industry, limit)
        
        # Analyze reports
        analyzer = TrendAnalyzer()
        analysis = analyzer.analyze_industry_reports(reports, industry)
        
        return jsonify({
            'reports': reports,
            'analysis': analysis
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error scraping industry reports: {str(e)}'
        }), 500

@api_bp.route('/trend-detector/economic-indicators', methods=['GET'])
@jwt_required()
def get_economic_indicators():
    """Get current economic indicators"""
    try:
        # Initialize scraper
        news_api_key = current_app.config.get('NEWS_API_KEY')
        twitter_api_key = current_app.config.get('TWITTER_API_KEY')
        scraper = TrendScraper(news_api_key, twitter_api_key)
        
        # Scrape economic indicators
        indicators = scraper.scrape_economic_indicators()
        
        # Analyze indicators
        analyzer = TrendAnalyzer()
        analysis = analyzer.analyze_economic_indicators(indicators)
        
        return jsonify({
            'indicators': indicators,
            'analysis': analysis
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error getting economic indicators: {str(e)}'
        }), 500

@api_bp.route('/trend-detector/comprehensive-report', methods=['POST'])
@jwt_required()
def generate_trend_report():
    """Generate comprehensive trend report"""
    try:
        # Get parameters
        industry = request.json.get('industry')
        keywords = request.json.get('keywords', [])
        
        if not industry or not keywords:
            return jsonify({'error': 'Industry and keywords are required'}), 400
        
        # Initialize scraper and analyzer
        news_api_key = current_app.config.get('NEWS_API_KEY')
        twitter_api_key = current_app.config.get('TWITTER_API_KEY')
        scraper = TrendScraper(news_api_key, twitter_api_key)
        analyzer = TrendAnalyzer()
        
        # Collect data
        articles = scraper.scrape_news_articles(keywords, 7, 100)
        social_data = scraper.scrape_social_media_trends(keywords, 7)
        reports = scraper.scrape_industry_reports(industry, 5)
        indicators = scraper.scrape_economic_indicators()
        
        # Analyze data
        news_analysis = analyzer.analyze_news_articles(articles)
        social_analysis = analyzer.analyze_social_trends(social_data)
        industry_analysis = analyzer.analyze_industry_reports(reports, industry)
        economic_analysis = analyzer.analyze_economic_indicators(indicators)
        
        # Generate comprehensive report
        report = analyzer.generate_trend_report(
            news_analysis, social_analysis, industry_analysis, economic_analysis, industry
        )
        
        return jsonify({
            'report': report
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating trend report: {str(e)}'
        }), 500

# Sustainability and Ethics Routes
@api_bp.route('/sustainability/ethical-analysis', methods=['POST'])
@jwt_required()
def analyze_business_ethics():
    """Analyze business practices for ethical considerations"""
    try:
        # Get parameters
        business_data = request.json.get('business_data', {})
        
        # Initialize analyzer
        api_key = current_app.config.get('OPENAI_API_KEY')
        analyzer = EthicalAnalyzer(api_key)
        
        # Analyze business practices
        result = analyzer.analyze_business_practices(business_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error analyzing business ethics: {str(e)}'
        }), 500

@api_bp.route('/sustainability/supply-chain-analysis', methods=['POST'])
@jwt_required()
def analyze_supply_chain():
    """Analyze supply chain for ethical considerations"""
    try:
        # Get parameters
        supply_chain_data = request.json.get('supply_chain_data', {})
        
        # Initialize analyzer
        api_key = current_app.config.get('OPENAI_API_KEY')
        analyzer = EthicalAnalyzer(api_key)
        
        # Analyze supply chain
        result = analyzer.analyze_supply_chain(supply_chain_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error analyzing supply chain: {str(e)}'
        }), 500

@api_bp.route('/sustainability/environmental-metrics', methods=['POST'])
@jwt_required()
def calculate_environmental_metrics():
    """Calculate environmental sustainability metrics"""
    try:
        # Get parameters
        environmental_data = request.json.get('environmental_data', {})
        revenue = request.json.get('revenue', 1000000)
        industry = request.json.get('industry', 'default')
        
        # Initialize metrics calculator
        metrics_calculator = SustainabilityMetrics()
        
        # Calculate metrics
        result = metrics_calculator.calculate_environmental_metrics(
            environmental_data, revenue, industry
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error calculating environmental metrics: {str(e)}'
        }), 500

@api_bp.route('/sustainability/social-metrics', methods=['POST'])
@jwt_required()
def calculate_social_metrics():
    """Calculate social sustainability metrics"""
    try:
        # Get parameters
        social_data = request.json.get('social_data', {})
        industry = request.json.get('industry', 'default')
        
        # Initialize metrics calculator
        metrics_calculator = SustainabilityMetrics()
        
        # Calculate metrics
        result = metrics_calculator.calculate_social_metrics(social_data, industry)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error calculating social metrics: {str(e)}'
        }), 500

@api_bp.route('/sustainability/governance-metrics', methods=['POST'])
@jwt_required()
def calculate_governance_metrics():
    """Calculate governance sustainability metrics"""
    try:
        # Get parameters
        governance_data = request.json.get('governance_data', {})
        
        # Initialize metrics calculator
        metrics_calculator = SustainabilityMetrics()
        
        # Calculate metrics
        result = metrics_calculator.calculate_governance_metrics(governance_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error calculating governance metrics: {str(e)}'
        }), 500

@api_bp.route('/sustainability/esg-score', methods=['POST'])
@jwt_required()
def calculate_esg_score():
    """Calculate overall ESG score"""
    try:
        # Get parameters
        environmental_metrics = request.json.get('environmental_metrics', {})
        social_metrics = request.json.get('social_metrics', {})
        governance_metrics = request.json.get('governance_metrics', {})
        
        # Initialize metrics calculator
        metrics_calculator = SustainabilityMetrics()
        
        # Calculate ESG score
        result = metrics_calculator.calculate_esg_score(
            environmental_metrics, social_metrics, governance_metrics
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error calculating ESG score: {str(e)}'
        }), 500

@api_bp.route('/sustainability/report', methods=['POST'])
@jwt_required()
def generate_sustainability_report():
    """Generate sustainability report"""
    try:
        # Get parameters
        business_data = request.json.get('business_data', {})
        environmental_data = request.json.get('environmental_data', {})
        social_data = request.json.get('social_data', {})
        
        # Initialize analyzer
        api_key = current_app.config.get('OPENAI_API_KEY')
        analyzer = EthicalAnalyzer(api_key)
        
        # Generate report
        result = analyzer.generate_sustainability_report(
            business_data, environmental_data, social_data
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating sustainability report: {str(e)}'
        }), 500

# Risk Manager Routes
@api_bp.route('/risk-manager/financial-risks', methods=['POST'])
@jwt_required()
def predict_financial_risks():
    """Predict financial risks"""
    try:
        # Get parameters
        financial_data = request.json.get('financial_data', {})
        
        # Initialize risk predictor
        model_path = current_app.config.get('RISK_MODEL_PATH')
        predictor = RiskPredictor(model_path)
        
        # Predict risks
        result = predictor.predict_financial_risks(financial_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error predicting financial risks: {str(e)}'
        }), 500

@api_bp.route('/risk-manager/market-risks', methods=['POST'])
@jwt_required()
def predict_market_risks():
    """Predict market risks"""
    try:
        # Get parameters
        market_data = request.json.get('market_data', {})
        
        # Initialize risk predictor
        predictor = RiskPredictor()
        
        # Predict risks
        result = predictor.predict_market_risks(market_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error predicting market risks: {str(e)}'
        }), 500

@api_bp.route('/risk-manager/operational-risks', methods=['POST'])
@jwt_required()
def predict_operational_risks():
    """Predict operational risks"""
    try:
        # Get parameters
        operational_data = request.json.get('operational_data', {})
        
        # Initialize risk predictor
        predictor = RiskPredictor()
        
        # Predict risks
        result = predictor.predict_operational_risks(operational_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error predicting operational risks: {str(e)}'
        }), 500

@api_bp.route('/risk-manager/monte-carlo', methods=['POST'])
@jwt_required()
def run_risk_monte_carlo():
    """Run Monte Carlo simulation for risk assessment"""
    try:
        # Get parameters
        simulation_data = request.json.get('simulation_data', {})
        scenarios = request.json.get('scenarios', 1000)
        
        # Initialize risk predictor
        predictor = RiskPredictor()
        
        # Run simulation
        result = predictor.monte_carlo_simulation(simulation_data, scenarios)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error running Monte Carlo simulation: {str(e)}'
        }), 500

@api_bp.route('/risk-manager/mitigation-strategies', methods=['POST'])
@jwt_required()
def get_mitigation_strategies():
    """Get risk mitigation strategies"""
    try:
        # Get parameters
        risk_data = request.json.get('risk_data', {})
        
        # Initialize mitigation strategies
        strategies = MitigationStrategies()
        
        # Get strategies
        result = strategies.get_mitigation_strategies(risk_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error getting mitigation strategies: {str(e)}'
        }), 500

@api_bp.route('/risk-manager/custom-strategy', methods=['POST'])
@jwt_required()
def generate_custom_mitigation_strategy():
    """Generate custom risk mitigation strategy"""
    try:
        # Get parameters
        risk_type = request.json.get('risk_type', 'financial_risk')
        risk_factors = request.json.get('risk_factors', [])
        business_constraints = request.json.get('business_constraints', {})
        
        # Initialize mitigation strategies
        strategies = MitigationStrategies()
        
        # Generate custom strategy
        result = strategies.generate_custom_strategy(risk_type, risk_factors, business_constraints)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating custom mitigation strategy: {str(e)}'
        }), 500

@api_bp.route('/recommender/growth-strategies', methods=['POST'])
@jwt_required()
def get_growth_recommendations():
    """Get growth strategy recommendations"""
    try:
        # Get parameters
        business_data = request.json.get('business_data', {})
        market_data = request.json.get('market_data', {})
        trend_data = request.json.get('trend_data', {})
        
        # Initialize recommender
        recommender = BusinessRecommender()
        
        # Set data
        recommender.set_business_data(business_data)
        recommender.set_market_data(market_data)
        if trend_data:
            recommender.set_trend_data(trend_data)
        
        # Generate recommendations
        result = recommender.generate_growth_recommendations()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating growth recommendations: {str(e)}'
        }), 500

@api_bp.route('/recommender/competitive-strategy', methods=['POST'])
@jwt_required()
def get_competitive_strategy():
    """Get competitive strategy recommendations"""
    try:
        # Get parameters
        business_data = request.json.get('business_data', {})
        market_data = request.json.get('market_data', {})
        
        # Initialize recommender
        recommender = BusinessRecommender()
        
        # Set data
        recommender.set_business_data(business_data)
        recommender.set_market_data(market_data)
        
        # Generate competitive strategy
        result = recommender.generate_competitive_strategy()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating competitive strategy: {str(e)}'
        }), 500

@api_bp.route('/recommender/tailored-strategy', methods=['POST'])
@jwt_required()
def get_tailored_growth_strategy():
    """Get tailored growth strategy"""
    try:
        # Get parameters
        business_data = request.json.get('business_data', {})
        market_data = request.json.get('market_data', {})
        growth_preferences = request.json.get('growth_preferences', {})
        
        # Initialize growth strategies
        strategies = GrowthStrategies()
        
        # Generate tailored strategy
        result = strategies.generate_tailored_strategy(
            business_data, market_data, growth_preferences
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error generating tailored growth strategy: {str(e)}'
        }), 500

@api_bp.route('/tasks/<task_id>', methods=['GET'])
@jwt_required()
def get_task_status(task_id):
    """Get status of an asynchronous task"""
    try:
        from app import celery
        
        # Get task result
        task = celery.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'status': 'Pending...'
            }
        elif task.state == 'FAILURE':
            response = {
                'state': task.state,
                'status': 'Failed',
                'error': str(task.info)
            }
        else:
            response = {
                'state': task.state,
                'status': task.info.get('status', '') if task.info else ''
            }
            
            # Add task-specific info if available
            if task.info and isinstance(task.info, dict):
                for key, value in task.info.items():
                    if key != 'status':
                        response[key] = value
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error getting task status: {str(e)}'
        }), 500

# Error handlers
@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0"
    }), 200