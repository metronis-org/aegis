#!/usr/bin/env python
'''Initialize database with migrations.'''

import subprocess
import sys
import os

def main():
    print('[INFO] Running database migrations...')

    # Run alembic upgrade
    result = subprocess.run(
        ['python', '-m', 'alembic', 'upgrade', 'head'],
        env={**os.environ, 'PYTHONPATH': 'src'},
    )

    if result.returncode == 0:
        print('[OK] Database initialized successfully!')
    else:
        print('[ERROR] Database initialization failed')
        sys.exit(1)

if __name__ == '__main__':
    main()
