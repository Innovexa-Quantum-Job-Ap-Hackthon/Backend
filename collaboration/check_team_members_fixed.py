from Database.DBConfiguration.database import SessionLocal
from Database.DBmodels.database_models import TeamMember
from Database.DBmodels.extended_database_models import Role, Permission

db = SessionLocal()
try:
    # Check team members for team 9
    team_members = db.query(TeamMember).filter(TeamMember.team_id == 9).all()
    print(f'Team 9 members: {len(team_members)}')
    for member in team_members:
        print(f'  User ID: {member.user_id}, Team ID: {member.team_id}')

    # Check if user 1 is in team 9
    user_1_in_team_9 = db.query(TeamMember).filter(TeamMember.team_id == 9, TeamMember.user_id == 1).first()
    print(f'User 1 in team 9: {user_1_in_team_9 is not None}')

    # Check all teams user 1 is in
    user_1_teams = db.query(TeamMember).filter(TeamMember.user_id == 1).all()
    print(f'User 1 teams: {len(user_1_teams)}')
    for team in user_1_teams:
        print(f'  Team ID: {team.team_id}')

finally:
    db.close()
