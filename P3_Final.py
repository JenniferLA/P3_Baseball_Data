# Import Libraries 
import pandas as pd
import csv
import sqlite3
import matplotlib.pyplot as plt


# Connection to SQLite Baseball db
conn = sqlite3.connect('Baseball.db')
print('DB Init')
cursor = conn.cursor()


## - Creating Tables: 
# Create People Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS People (
    playerID TEXT PRIMARY KEY,
    birthYear INTEGER,
    birthMonth INTEGER,
    birthDay INTEGER,
    birthCountry TEXT,
    birthState TEXT,
    birthCity TEXT,
    deathYear INTEGER,
    deathMonth INTEGER,
    deathDay INTEGER,
    deathCountry TEXT,
    deathState TEXT,
    deathCity TEXT,
    nameFirst TEXT,
    nameLast TEXT,
    nameGiven TEXT,
    weight INTEGER,
    height INTEGER,
    bats TEXT,
    throws TEXT,
    debut DATE,
    finalGame DATE,
    retroID TEXT,
    bbrefID                       
)
''')

# Create Batting Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Batting (
    playerID TEXT,
    yearID INTEGER,
    stint INTEGER,
    teamID TEXT,
    lgID TEXT,
    G INTEGER,
    AB INTEGER,
    R INTEGER,
    H INTEGER,
    "2B" INTEGER,
    "3B" INTEGER,           
    HR INTEGER,
    RBI INTEGER,
    SB INTEGER,
    CS INTEGER,
    BB INTEGER,
    SO INTEGER,
    IBB INTEGER,
    HBP INTEGER,
    SH INTEGER,
    SF INTEGER,
    GIDP INTEGER,                       
    PRIMARY KEY (playerID, yearID, stint),
    FOREIGN KEY (playerID) REFERENCES People(playerID)
)
''')

# Create the Teams Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Teams (
    yearID INTEGER,
    IgID TEXT,
    teamID TEXT,
    franchID TEXT,
    divID TEXT,
    Rank INTEGER,
    G INTEGER,
    Ghome INTEGER,
    W INTEGER,
    L INTEGER,
    DivWin TEXT,
    WCWin TEXT,
    LgWin TEXT, 
    WSWin TEXT, 
    R INTEGER,
    AB INTEGER,
    H INTEGER,
    "2B" INTEGER,
    "3B" INTEGER,
    HR INTEGER,
    BB INTEGER,
    SO INTEGER,
    SB INTEGER,
    CS INTEGER,
    HBP INTEGER,
    SF INTEGER,
    RA INTEGER,
    ER INTEGER,
    ERA REAL,
    CG INTEGER,
    SHO INTEGER,
    SV INTEGER,
    IPouts INTEGER,
    HA INTEGER,
    HRA INTEGER,
    BBA INTEGER,
    SOA INTEGER,
    E INTEGER,
    DP INTEGER,
    FP REAL,
    name TEXT,
    park TEXT, 
    attendance INTEGER,
    BPF INTEGER,
    PPF INTEGER,
    teamIDBR TEXT,
    teamIDlahman45 TEXT, 
    teamIDretro TEXT
)
''')

# Create the Pitching Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Pitching (
    playerID TEXT,
    yearID INTEGER,
    stint INTEGER,
    teamID TEXT,
    lgID TEXT,
    W INTEGER,
    L INTEGER,
    G INTEGER,
    GS INTEGER,
    CG INTEGER,
    SHO INTEGER,
    SV INTEGER,
    IPouts INTEGER,
    H INTEGER,
    ER INTEGER,
    HR INTEGER,
    BB INTEGER,
    SO INTEGER,
    BAOpp REAL,
    ERA REAL,
    IBB INTEGER,
    WP INTEGER,
    HBP INTEGER,
    BK INTEGER,
    BFP INTEGER,
    GF INTEGER,
    R INTEGER,
    SH INTEGER,
    SF INTEGER,
    GIDP INTEGER,
    PRIMARY KEY (playerID, yearID, stint)
    FOREIGN KEY (playerID) REFERENCES People(playerID)
    )
''')

# Create the Franchise Table 
cursor.execute('''
CREATE TABLE IF NOT EXISTS Franchises (
    franchID TEXT,
    franchName TEXT, 
    active TEXT,
    NAassoc TEXT
)
''')


## - Reading CSV Files and Inserting Data:
# Read People CSV File and Insert Data into People Table
with open ('/Users/jp/Desktop/Baseball.db/People.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        cursor.execute('''
        INSERT OR IGNORE INTO People (
          playerID, birthYear, birthMonth, birthDay, birthCountry, birthState, birthCity,
                    deathYear, deathMonth, deathDay, deathCountry, deathState, deathCity,
                    nameFirst, nameLast, nameGiven, weight, height, bats, throws, debut, finalGame, retroID, bbrefID
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', row)

# Read Batting CSV File and Insert Data into Batting Table 
with open ('/Users/jp/Desktop/Baseball.db/Batting.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        cursor.execute('''
        INSERT OR IGNORE INTO Batting (
          playerID, yearID, stint, teamID, lgID, G, AB, R, H, "2B", "3B", HR, RBI, SB, CS, BB, SO, IBB, HBP, SH, SF, GIDP
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', row)

# Read Pitching CSV file and Insert Data into Pitching Table
with open ('/Users/jp/Desktop/Baseball.db/Pitching.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        cursor.execute('''
        INSERT OR IGNORE INTO Pitching (
          playerID, yearID, stint, teamID, lgID, W, L, G, GS, CG, SHO, SV, IPOuts, H, ER,
                    HR, BB, SO, BAOpp, ERA, IBB, WP, HBP, BK, BFP, GF, R, SH, SF, GIDP
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)       
        ''', row)

# Read Teams CSV File and Insert Data into Teams Table
with open ('/Users/jp/Desktop/Baseball.db/Teams.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        cursor.execute('''
        INSERT OR IGNORE INTO Teams (
          yearID, IgID, teamID, franchID, divID, Rank, G, Ghome, W, L, DivWin, WCWin, LgWin, WSWin, R, AB, H, "2B", "3B",
                    HR, BB, SO, SB, CS, HBP, SF, RA, ER, ERA, CG, SHO, SV, IPouts, HA, HRA, BBA, SOA, E, DP, FP, name,
                    park, attendance, BPF, PPF, teamIDBR, teamIDlahman45, teamIDretro
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)       
        ''', row)

# Read TeamFranchises CSV File and Insert Data into Franchises Table
with open ('/Users/jp/Desktop/Baseball.db/TeamsFranchises.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        cursor.execute('''
        INSERT INTO Franchises (franchID, franchName, active, NAassoc)
        VALUES (?, ?, ?, ?)
        ''', row)
# Commit the Changes 
conn.commit()


## - Data Cleaning:
# Define the SQL Query
query = """
SELECT 
    p.playerID, p.nameFirst, p.nameLast, p.nameGiven, 
    p.birthYear, p.birthMonth, p.birthDay, 
    p.birthCountry, p.birthState, p.birthCity, 
    p.weight, p.throws, p.bats, p.deathYear,  
    b.*  -- All columns from the Batting table
FROM 
    People p
JOIN 
    Batting b ON p.playerID = b.playerID
WHERE
    p.finalGame > '2021-10-02'
"""    
    
    
"""
    b.G >=50 AND p.finalGame > '2021-10-02'; 
"""

# Execute the Query
cursor.execute(query)

# Fetch All Results
results = cursor.fetchall()

# Column Names from the Cursor Description
column_names = [description[0] for description in cursor.description]

# Convert the Data to Pandas DataFrame
df = pd.DataFrame(results, columns=column_names)

# Display the DataFrame
print(df)

# Print Column Names
#print("Columns in Dataframe:", df.columns.tolist())

# Convert 'birthYear' Column to Numeric (int)
df['birthYear'] = pd.to_numeric(df['birthYear'], errors='coerce')

# Add Calculated Column for Player's Age
df['age'] = 2024 - df['birthYear']

# Add Calculated Column for Player's Full Name (first + last name)
df['fullName'] = df['nameFirst'] + ' ' + df['nameLast']

# Compare the columns
#print("Columns in Dataframe:", df.columns.tolist())

# Drop Unnecessary Birth Date and Name-Related Columns
columns_to_drop = ['nameFirst', 'nameLast', 'nameGiven',
                   'birthYear', 'birthMonth',
                   'birthCountry', 'birthState', 'birthCity', 
                   'deathYear', 'birthDay']

#df = df.drop(columns=columns_to_drop)
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Display the DF with the New Columns
print(df.head())

# Drop Rows with Any Missing Values
df_cleaned = df.dropna

# Display the Cleaned DF
print(df_cleaned())


## - Required Questions:
# Which active player had the most runs batted in (“RBI” from the Batting table) from 2015-2018?
query = """
SELECT 
    p.playerID,
    p.nameFirst,
    p.nameLast,
    SUM(b.RBI) as total_RBI
FROM 
    People p
JOIN 
    Batting b ON p.playerID = b.playerID
WHERE
    b.yearID BETWEEN 2015 AND 2018 
GROUP BY
    p.playerID, p.nameFirst, p.nameLast
ORDER BY
    total_RBI DESC
LIMIT 1;
"""

# Execute and Fetch 
cursor.execute(query)
result = cursor.fetchone()

# Display Result
if result:
    print(f"The active player with the most RBIs from 2015-2018 is {result[1]} {result[2]} with {result[3]} RBIs.")
else:
    print("No active players found.")


# How many double plays did Albert Pujols ground into (“GIDP” from Batting table) in 2016?
query = """
SELECT 
    SUM(b.GIDP) as total_GIDP
FROM
    Batting b
JOIN
    People p ON b.playerID = p.playerID
WHERE
    p.nameFirst = 'Albert'
    AND p.nameLast = 'Pujols'
    AND b.yearID = 2016;
"""

# Execute and Fetch
cursor.execute(query)
result = cursor.fetchone()

# Display the Result
if result and result[0] is not None:
    print(f"Albert Pujols grounded into {result[0]} double plays in 2016.")
else:
    print("No data found for Albert Pujols in 2016.")



## - Create Various Plots:
# 1. Histogram of Triples (3B) per Year
query = """
SELECT 
    yearID,
    sum(b."3B") as total_triples
FROM
    Batting b
GROUP BY
    yearID
ORDER BY
    yearID;
"""

# Execute and Fetch
cursor.execute(query)
results = cursor.fetchall()

# Convert to pd Dataframe
df = pd.DataFrame(results, columns=['yearID', 'total_triples'])

# Create Histogram
plt.figure(figsize=(10,6))
plt.bar(df['yearID'], df['total_triples'], color = 'skyblue')
plt.title('Total Triples (3B) Per Year')
plt.xlabel('Year')
plt.ylabel('Total Triples')
plt.grid(True)
# Display the histogram
plt.show()


# 2. Scatter Plot Relating Triples (3B) and Steals (SB)
query = """
SELECT 
    playerID,
    yearID,
    SUM(b."3B") as total_triples,
    SUM(b.SB) as total_steals
FROM 
    Batting b
Group By
    playerID, yearID
HAVING
    total_triples > 0 AND total_steals > 0
ORDER BY
    yearID;
"""

# Execute and Fetch
cursor.execute(query)
results = cursor.fetchall()

# Convert to pd Dataframe
df = pd.DataFrame(results, columns=['playerID', 'yearID', 'total_triples', 'total_steals'])

# Create the Scatter Plot
plt.figure(figsize=(10,6))
plt.scatter(df['total_triples'], df['total_steals'], color = 'blue', alpha=0.6)
plt.title('Scatter Plot of Triples (3B) vs. Steals (SB)')
plt.xlabel('Total Triples (3B)')
plt.ylabel('Total Steals (SB)')
plt.grid(True)
# Display the scatter plot
plt.show()


# 3. Stacked Area Plot of Top Home Run Hitters Over Time
query = """
SELECT
    p.nameFirst,
    p.nameLast,
    b.yearID,
    SUM(b.HR) as total_HR
FROM 
    Batting b
JOIN 
    People p ON b.playerID = p.playerID
GROUP BY
    p.nameFirst, p.nameLast, b.yearID
ORDER BY
    total_HR DESC
LIMIT 10;
"""

# Execute and Fetch
cursor.execute(query)
results = cursor.fetchall()

# Convert to DF
df = pd.DataFrame(results, columns=['nameFirst', 'nameLast', 'yearID', 'total_HR'])
# Combine first and last names 
df['playerName'] = df['nameFirst'] + ' ' + df['nameLast']

# Pivot the Data to Get Years as the Index and Players as Columns
df_pivot = df.pivot(index='yearID', columns='playerName', values='total_HR').fillna(0)

# Plot
plt.figure(figsize=(12, 6))
df_pivot.plot(kind='area', stacked=True, alpha=0.5, colormap='tab20', figsize=(12, 6))
plt.title('Top Home Run Hitters Over Time (Stacked Area Plot)')
plt.xlabel('Year')
plt.ylabel('Total Home Runs')
plt.legend(title='Player', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# 4. Bar Graph of Career RBI Leaders
query = """
SELECT 
    p.nameFirst,
    p.nameLast, 
    SUM(b.RBI) as total_RBI
FROM 
    Batting b
JOIN 
    People p ON b.playerID = p.playerID
GROUP BY
    p.playerID
ORDER BY 
    total_RBI DESC
LIMIT 10;
"""

# Execute and Fetch
cursor.execute(query)
results = cursor.fetchall()

# Convert to DF
df = pd.DataFrame(results, columns=['nameFirst', 'nameLast', 'total_RBI'])

# Combine First and Last Names 
df['playerName'] = df['nameFirst'] + ' ' + df['nameLast']
# Ensure batting_average is numeric
#df['batting_average'] = pd.to_numeric(df['batting_average'])

# Plot
plt.figure(figsize=(12, 6))
plt.barh(df['playerName'], df['total_RBI'], color='blue')
plt.title('Career RBI Leaders')
plt.xlabel('Total RBI')
plt.ylabel('Player')
plt.grid(True)
plt.show()


# 5. Bar Graph of Player Longevity: Top 10 Longest Careers  
query = """
SELECT 
    p.nameFirst,
    p.nameLast,
    MIN(b.yearID) as debut_year,
    MAX(b.yearID) as final_year,
    (MAX(b.yearID) - MIN(b.yearID) + 1) as career_span
FROM 
    Batting b
JOIN 
    People p ON b.playerID = p.playerID
GROUP BY 
    p.playerID
ORDER BY
    career_span DESC
LIMIT 10;
"""

# Execute and Fetch
cursor.execute(query)
results = cursor.fetchall()

# Convert to DataFrame
df = pd.DataFrame(results, columns=['nameFirst', 'nameLast', 'debut_year', 'final_year', 'career_span'])
# Combine first and last names
df['playerName'] = df['nameFirst'] + ' ' + df['nameLast']

# Plot
plt.figure(figsize=(12, 6))
plt.barh(df['playerName'], df['career_span'], color='purple')
plt.title('Player Longevity (Top 10 Longest Careers)')
plt.xlabel('Career Span (Years)')
plt.ylabel('Player')
plt.grid(True)
plt.show()


# 6. Scatter Plot Player Weight vs. Performance
query = """
SELECT 
    p.weight, 
    AVG(b.HR) as avg_HR
FROM 
    Batting b
JOIN 
    People p ON b.playerID = p.playerID
GROUP BY 
    p.weight
HAVING 
    COUNT(b.playerID) > 100 
ORDER BY 
    p.weight;
"""

# Execute and Fetch
cursor.execute(query)
results = cursor.fetchall()

# Convert to DF
df = pd.DataFrame(results, columns=['weight', 'avg_HR'])
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

# Plot 
plt.figure(figsize=(12, 6))
plt.scatter(df['weight'], df['avg_HR'], color='green')
plt.title('Player Weight vs. Average Home Runs')
plt.xlabel('Weight (lbs)')
plt.ylabel('Average Home Runs')
plt.grid(True)
plt.show()


# 7. Graph Pitching VS. Batting Performance Over Time 
query = """
SELECT
    b.yearID,
    SUM(b.HR) as total_HR,
    SUM(p.SO) as total_SO
FROM 
    Batting b
JOIN 
    Pitching p ON b.yearID = p.yearID
GROUP BY
    b.yearID
ORDER BY
    b.yearID;
"""

cursor.execute(query)
results = cursor.fetchall()

# Convert to DF
df = pd.DataFrame(results, columns=['yearID', 'total_HR', 'total_SO'])

# Plot
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Year')
ax1.set_ylabel('Total Home Runs', color='tab:blue')
ax1.plot(df['yearID'], df['total_HR'], color='tab:blue', label='Total Home Runs')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Total Strikeouts', color='tab:red')
ax2.plot(df['yearID'], df['total_SO'], color='tab:red', label='Total Strikeouts')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Pitching vs.Batting Performance Over Time')
fig.tight_layout()
plt.grid(True)
plt.show()


# 8. Graph Strikeouts vs. Walks (Pitching)
query = """
SELECT 
    p.yearID,
    SUM(p.SO) as total_strikeouts,
    SUM(p.BB) as total_walks
FROM 
    Pitching p
GROUP BY 
    p.yearID
ORDER BY 
    p.yearID;
"""

# Execute and Fetch
cursor.execute(query)
results = cursor.fetchall()

# Convert to DataFrame
df = pd.DataFrame(results, columns=['yearID', 'total_strikeouts', 'total_walks'])
# Combine the first and last names
#df['playerName'] = df['nameFirst'] + ' ' + df['nameLast']

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['yearID'], df['total_strikeouts'], label='Total Strikeouts', color='red')
plt.plot(df['yearID'], df['total_walks'], label="Total Walks", color='blue')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Strikeouts vs. Walks (Pitching)')
plt.legend()
plt.grid(True)
plt.show()


# 9. Graph - Top 10 Team Performance Over Time
query = """
WITH TeamWins AS (
    SELECT
        t.name,
        t.yearID, 
        t.W as wins 
    FROM 
        Teams t
),
TopTeams AS (
    SELECT
        name, 
        SUM(wins) as total_wins
    FROM 
        TeamWins
    GROUP BY
        name
    ORDER BY
        total_wins DESC
    LIMIT 10
)
SELECT
    tw.name,
    tw.yearID,
    tw.wins
FROM
    TeamWins tw
JOIN 
    TopTeams tt ON tw.name = tt.name
ORDER BY
    tw.name, tw.yearID;
"""

# Execute and Fetch 
cursor.execute(query)
results = cursor.fetchall()

# Convert to DataFrame 
df = pd.DataFrame(results, columns=['teamName', 'yearID', 'wins'])

# Plot
plt.figure(figsize=(12,6))
for team in df['teamName']. unique():
    team_data = df[df['teamName'] == team]
    plt.plot(team_data['yearID'], team_data['wins'], label=team)

plt.title('Top 10 Team Performance Over Time')
plt.xlabel('Year')
plt.ylabel('Wins')
plt.legend(title='Team', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# 10. Bar Graph Home Runs by Ballpark
query = """
SELECT 
    t.park, 
    t.yearID,
    sum(t.HR) as total_HR
FROM
    Teams t
GROUP BY
    t.park, t.yearID
ORDER BY
    total_HR DESC
LIMIT 10;
"""

# Execute and Fetch
cursor.execute(query)
results = cursor.fetchall()

# Convert 
df = pd.DataFrame(results, columns=['park', 'yearID', 'total_HR'])

# Plot
plt.figure(figsize=(12, 6))
plt.barh(df['park'], df['total_HR'], color='orange')
plt.title('Home Runs by Ballpark (Top 10)')
plt.xlabel('Total Home Runs')
plt.ylabel('Ballpark')
plt.grid(True)
plt.show()

# Close the Connection
conn.close()