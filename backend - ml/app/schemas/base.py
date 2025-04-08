from marshmallow import Schema, fields, validate

class BaseSchema(Schema):
    """Base schema class with common functionality"""
    id = fields.Int(dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    def handle_error(self, error, data, **kwargs):
        """Handle validation errors"""
        raise ValueError(error.messages) 