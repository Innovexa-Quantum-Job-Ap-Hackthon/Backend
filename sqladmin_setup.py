from fastapi import FastAPI
from sqladmin import Admin, ModelView
from Database.DBConfiguration.database import engine, Base, SessionLocal
import Database.DBmodels.database_models as models

# Import the main FastAPI app
from main import app

# Define a ModelView for each model you want to manage
class UserAdmin(ModelView, model=models.User):
    column_list = [models.User.id, models.User.email, models.User.name, models.User.is_active]
    column_searchable_list = [models.User.email, models.User.name]
    column_sortable_list = [models.User.id, models.User.email]
    can_create = True
    can_edit = True
    can_delete = True

class TeamAdmin(ModelView, model=models.Team):
    column_list = [models.Team.id, models.Team.name, models.Team.description, models.Team.created_at]
    can_create = True
    can_edit = True
    can_delete = True

class JobLogsAdmin(ModelView, model=models.JobLogs):
    column_list = [models.JobLogs.jobId, models.JobLogs.Device, models.JobLogs.Status, models.JobLogs.Shots, models.JobLogs.JobRasied, models.JobLogs.JobCompletion]
    can_create = False
    can_edit = True
    can_delete = True

# Initialize SqlAdmin
admin = Admin(app, engine, prefix="/admin")

# Add views
admin.add_view(UserAdmin)
admin.add_view(TeamAdmin)
admin.add_view(JobLogsAdmin)

# You can add more ModelViews for other models as needed
