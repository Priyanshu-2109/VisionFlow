from app.models.base import BaseModel
from app import db

class Role(BaseModel):
    __tablename__ = 'roles'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.String(200))
    permissions = db.Column(db.JSON)
    
    def __init__(self, name, description=None, permissions=None):
        self.name = name
        self.description = description
        self.permissions = permissions or {}
    
    def has_permission(self, permission):
        """Check if role has specific permission"""
        return permission in self.permissions.get('permissions', [])
    
    def add_permission(self, permission):
        """Add a permission to the role"""
        if 'permissions' not in self.permissions:
            self.permissions['permissions'] = []
        if permission not in self.permissions['permissions']:
            self.permissions['permissions'].append(permission)
    
    def remove_permission(self, permission):
        """Remove a permission from the role"""
        if permission in self.permissions.get('permissions', []):
            self.permissions['permissions'].remove(permission)
    
    @classmethod
    def get_by_name(cls, name):
        """Get role by name"""
        return cls.query.filter_by(name=name).first()
    
    @classmethod
    def create_default_roles(cls):
        """Create default roles if they don't exist"""
        default_roles = {
            'admin': {
                'description': 'Administrator with full access',
                'permissions': ['*']  # All permissions
            },
            'user': {
                'description': 'Regular user with basic access',
                'permissions': ['read:datasets', 'write:datasets', 'read:dashboards']
            },
            'viewer': {
                'description': 'View-only user',
                'permissions': ['read:datasets', 'read:dashboards']
            }
        }
        
        for role_name, role_data in default_roles.items():
            if not cls.get_by_name(role_name):
                cls(
                    name=role_name,
                    description=role_data['description'],
                    permissions=role_data['permissions']
                ).save() 