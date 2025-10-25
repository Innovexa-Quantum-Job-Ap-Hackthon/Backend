#!/usr/bin/env python3
"""
Database migration script to add missing columns to existing tables.
This script handles schema updates for SQLite database.
"""

import sqlite3
import os
from pathlib import Path

def migrate_database():
    """Add missing columns to existing database tables"""

    # Database path
    db_path = Path(__file__).parent.parent / "quantum.db"
    print(f"Looking for database at: {db_path}")
    print(f"Database exists: {db_path.exists()}")

    if not db_path.exists():
        print(f"Database file not found at {db_path}")
        return

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check if role_id column exists in users table
        cursor.execute("PRAGMA table_info('users')")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        print("Current users table columns:", column_names)

        # Add role_id column if it doesn't exist
        if 'role_id' not in column_names:
            print("Adding role_id column to users table...")
            cursor.execute("ALTER TABLE users ADD COLUMN role_id INTEGER REFERENCES roles(id)")
            print("✅ Added role_id column to users table")
        else:
            print("✅ role_id column already exists in users table")

        # Check if project_id column exists in JobLogs table
        cursor.execute("PRAGMA table_info('JobLogs')")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        print("Current JobLogs table columns:", column_names)

        # Add project_id column if it doesn't exist
        if 'project_id' not in column_names:
            print("Adding project_id column to JobLogs table...")
            cursor.execute("ALTER TABLE JobLogs ADD COLUMN project_id INTEGER REFERENCES projects(id)")
            print("✅ Added project_id column to JobLogs table")
        else:
            print("✅ project_id column already exists in JobLogs table")

        # Check if other potentially missing columns exist
        missing_columns = []

        # Check for user_id column
        if 'user_id' not in column_names:
            missing_columns.append(('user_id', 'INTEGER'))

        # Add any missing columns
        for col_name, col_type in missing_columns:
            print(f"Adding {col_name} column to JobLogs table...")
            cursor.execute(f"ALTER TABLE JobLogs ADD COLUMN {col_name} {col_type}")
            print(f"✅ Added {col_name} column to JobLogs table")

        # Commit changes
        conn.commit()
        print("✅ Database migration completed successfully!")

        # Verify the changes
        cursor.execute("PRAGMA table_info('JobLogs')")
        updated_columns = cursor.fetchall()
        print("\nUpdated JobLogs table columns:")
        for col in updated_columns:
            print(f"  - {col[1]} ({col[2]})")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    migrate_database()
