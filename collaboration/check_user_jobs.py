from Database.DBConfiguration.database import SessionLocal
from Database.DBmodels.database_models import JobLogs, User, TeamMember
from Database.DBmodels.extended_database_models import Role, Permission

db = SessionLocal()

try:
    # Check jobs for user 2
    jobs_user_2 = db.query(JobLogs).filter(JobLogs.user_id == 2).all()
    print(f"Jobs for user 2: {len(jobs_user_2)}")
    for job in jobs_user_2:
        print(f"  {job.jobId}: {job.Status} on {job.Device}")

    # Check all jobs
    all_jobs = db.query(JobLogs).all()
    print(f"\nTotal jobs in database: {len(all_jobs)}")
    for job in all_jobs:
        print(f"  {job.jobId}: User {job.user_id}, Status {job.Status}")

    # Check team members for team 10
    team_members = db.query(TeamMember).filter(TeamMember.team_id == 10).all()
    print(f"\nTeam 10 members: {len(team_members)}")
    for member in team_members:
        print(f"  User {member.user_id}: {member.role}")

    # Check if user 2 is in team 10
    user_2_in_team = db.query(TeamMember).filter(
        TeamMember.team_id == 10,
        TeamMember.user_id == 2
    ).first()
    print(f"\nUser 2 in team 10: {user_2_in_team is not None}")
    if user_2_in_team:
        print(f"  Role: {user_2_in_team.role}")

finally:
    db.close()
