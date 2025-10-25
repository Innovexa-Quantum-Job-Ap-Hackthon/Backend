from Database.DBConfiguration.database import SessionLocal
from Database.DBmodels.database_models import JobLogs
from Database.DBmodels.extended_database_models import Role, Permission

db = SessionLocal()
try:
    # Check jobs for user 1
    user_1_jobs = db.query(JobLogs).filter(JobLogs.user_id == 1).all()
    print(f'User 1 jobs: {len(user_1_jobs)}')
    for job in user_1_jobs:
        print(f'  Job ID: {job.jobId}, Device: {job.Device}, Status: {job.Status}, Created: {job.JobRasied}')

    # Check all jobs in the system
    all_jobs = db.query(JobLogs).all()
    print(f'Total jobs in system: {len(all_jobs)}')
    for job in all_jobs:
        print(f'  Job ID: {job.jobId}, User ID: {job.user_id}, Device: {job.Device}, Status: {job.Status}')

finally:
    db.close()
