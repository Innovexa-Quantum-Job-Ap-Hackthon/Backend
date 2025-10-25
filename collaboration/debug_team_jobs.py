from Database.DBConfiguration.database import SessionLocal
from Database.DBmodels.database_models import JobLogs, User, TeamMember
from Database.DBmodels.extended_database_models import Role, Permission
from datetime import datetime, timedelta

db = SessionLocal()

try:
    team_id = 10
    days = 30

    # Get team member user IDs
    member_ids = db.query(TeamMember.user_id).filter(
        TeamMember.team_id == team_id
    ).all()
    member_ids = [mid[0] for mid in member_ids]

    print(f"Team {team_id} member IDs: {member_ids}")

    # Get jobs from these users
    since_date = datetime.utcnow() - timedelta(days=days)
    print(f"Since date: {since_date}")

    # Check all jobs without date filter
    all_jobs_no_date = db.query(JobLogs).filter(
        JobLogs.user_id.in_(member_ids)
    ).all()
    print(f"All jobs for team members (no date filter): {len(all_jobs_no_date)}")

    # Check jobs with date filter
    all_jobs_with_date = db.query(JobLogs).filter(
        JobLogs.user_id.in_(member_ids),
        JobLogs.JobRasied >= since_date
    ).all()
    print(f"Jobs for team members (with date filter): {len(all_jobs_with_date)}")

    # Check job dates
    print("\nJob dates:")
    for job in all_jobs_no_date:
        print(f"  {job.jobId}: {job.JobRasied} (user {job.user_id})")

    # Check current time
    print(f"\nCurrent time: {datetime.utcnow()}")

finally:
    db.close()
