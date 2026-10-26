import os
from flask import Flask, render_template
from config import Config
from database.db import init_db
from services.model_service import get_model_status
from routes.dashboard_routes import dashboard_bp
from routes.prediction_routes import prediction_bp
from routes.data_routes import data_bp
from routes.analytics_routes import analytics_bp

def create_app(config_class=Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 1. Initialize SQLite Database
    try:
        init_db()
    except Exception as e:
        app.logger.error(f"Database initialization error: {e}")

    # 2. Register Blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(analytics_bp)

    # 3. Global Context Processor (makes model status available to sidebar layout)
    @app.context_processor
    def inject_global_status():
        try:
            status = get_model_status()
        except Exception:
            status = {'is_ready': False}
        return {'g_model_status': status}

    # 4. Graceful Error Handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template(
            'error.html',
            error_code=404,
            error_title="Page Not Found",
            error_subtitle="The requested URL could not be located on this server.",
            error_message="The page you were looking for does not exist, has been moved, or an invalid address was entered."
        ), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return render_template(
            'error.html',
            error_code=413,
            error_title="Payload Too Large",
            error_subtitle="The uploaded file exceeds maximum permitted limits.",
            error_message="The uploaded CSV file exceeds the application limit of 16MB. Please upload a smaller dataset file."
        ), 413

    @app.errorhandler(500)
    def internal_error(error):
        return render_template(
            'error.html',
            error_code=500,
            error_title="Internal Server Error",
            error_subtitle="An unexpected issue occurred while processing your request.",
            error_message="The server encountered an internal error. Please return to the dashboard or try again shortly."
        ), 500

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
