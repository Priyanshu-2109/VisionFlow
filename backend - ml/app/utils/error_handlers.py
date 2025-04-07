from flask import jsonify
from werkzeug.exceptions import HTTPException
from app import db

def register_error_handlers(app):
    """Register error handlers for the application"""
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors"""
        return jsonify({
            "message": "Bad Request",
            "error": str(error)
        }), 400

    @app.errorhandler(401)
    def unauthorized(error):
        """Handle 401 Unauthorized errors"""
        return jsonify({
            "message": "Unauthorized",
            "error": str(error)
        }), 401

    @app.errorhandler(403)
    def forbidden(error):
        """Handle 403 Forbidden errors"""
        return jsonify({
            "message": "Forbidden",
            "error": str(error)
        }), 403

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors"""
        return jsonify({
            "message": "Not Found",
            "error": str(error)
        }), 404

    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle 500 Internal Server errors"""
        db.session.rollback()
        return jsonify({
            "message": "Internal Server Error",
            "error": str(error)
        }), 500

    @app.errorhandler(HTTPException)
    def handle_exception(error):
        """Handle all HTTP exceptions"""
        return jsonify({
            "message": error.name,
            "error": str(error)
        }), error.code

    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        """Handle all other exceptions"""
        db.session.rollback()
        return jsonify({
            "message": "Internal Server Error",
            "error": str(error)
        }), 500 