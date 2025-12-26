#!/usr/bin/env python3
"""
Verify the database has the new physical_state column and correct ionization energy
"""

import sqlite3

conn = sqlite3.connect('research_inertia.db')
cursor = conn.cursor()

# Check schema
print("Database schema:")
cursor.execute("PRAGMA table_info(elements)")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col[1]} ({col[2]})")

print("\n" + "="*60)
print("Sample data:")
print("="*60)

# Check some sample elements
cursor.execute("""
    SELECT atomic_number, symbol, ionization_energy, physical_state
    FROM elements
    WHERE atomic_number IN (1, 6, 35, 80, 26)
    ORDER BY atomic_number
""")

results = cursor.fetchall()
for row in results:
    z, sym, ie, state = row
    print(f"{sym:3} (Z={z:3}): IE={ie:6.2f} eV, State={state}")

conn.close()

print("\nDatabase verification complete!")
