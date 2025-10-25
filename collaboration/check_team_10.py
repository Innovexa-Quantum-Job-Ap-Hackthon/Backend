from Database.DBConfiguration.database import SessionLocal
from Database.DBmodels.database_models import Team, TeamMember, JobLogs
from Database.DBmodels.extended_database_models import Role, Permission

db = SessionLocal()
try:
    # Check team 10 specifically
    team_10 = db.query(Team).filter(Team.id == 10).first()
    if team_10:
        print(f'Team 10 details: ID={team_10.id}, Name={team_10.name}, Created by={team_10.created_by_id}')
        team_10_members = db.query(TeamMember).filter(TeamMember.team_id == 10).all()
        print(f'Team 10 members: {len(team_10_members)}')
        for member in team_10_members:
            print(f'  User ID: {member.user_id}, Role: {member.role}')
    else:
        print('Team 10 not found')

    # Check all jobs in the system
    all_jobs = db.query(JobLogs).all()
    print(f'Total jobs in system: {len(all_jobs)}')
    for job in all_jobs:
        print(f'  Job ID: {job.jobId}, User ID: {job.user_id}, Device: {job.Device}, Status: {job.Status}')

    # Check jobs for team 10 members
    if team_10:
        team_10_members = db.query(TeamMember).filter(TeamMember.team_id == 10).all()
        user_ids = [member.user_id for member in team_10_members]
        if user_ids:
            team_jobs = db.query(JobLogs).filter(JobLogs.user_id.in_(user_ids)).all()
            print(f'Jobs for team 10 members: {len(team_jobs)}')
            for job in team_jobs:
                print(f'  Job ID: {job.jobId}, User ID: {job.user_id}, Device: {job.Device}, Status: {job.Status}')

finally:
    db.close()
