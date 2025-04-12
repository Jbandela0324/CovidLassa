import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Load dataset
df = pd.read_csv('time_series_covid19_deaths_US.csv')

# Drop unnecessary columns
df.drop(columns=['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Lat', 'Long_', 'Combined_Key', 'Admin2', 'Population'], inplace=True)

# Identify non-date columns
non_date_cols = ['Province_State', 'Country_Region']

# Identify date columns (all other columns)
date_cols = [col for col in df.columns if col not in non_date_cols]

# Convert column names to datetime format
new_col_names = {col: pd.to_datetime(col, format='%m/%d/%y', errors='coerce') for col in date_cols}

# Rename columns with converted datetime format
df.rename(columns=new_col_names, inplace=True)

# Remove NaT columns (if any failed conversion)
df = df.dropna(axis=1, how='all')

# Recalculate date columns after renaming
date_cols = [col for col in df.columns if col not in non_date_cols]

# Ensure there are still valid date columns
if not date_cols:
    raise ValueError("No valid date columns found after renaming.")

# Filter for US states
us_states = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 
     'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 
    'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 
    'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 
    'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 
    'West Virginia', 'Wisconsin', 'Wyoming'
]  

df = df[df['Province_State'].isin(us_states)]

# Sort data by state
df_sorted = df.sort_values(by=['Province_State'])

# Compute daily new deaths
df_new_deaths = df_sorted.set_index('Province_State')[date_cols].diff(axis=1).fillna(0)

# Reset index
df_new_deaths.reset_index(inplace=True)

# Melt dataset into long format
df_melted = df_new_deaths.melt(id_vars=['Province_State'], value_vars=date_cols, var_name='Date', value_name='New_Deaths')

# Ensure 'Date' is in datetime format before extracting the year
df_melted['Date'] = pd.to_datetime(df_melted['Date'])

# Extract year
df_melted['Year'] = df_melted['Date'].dt.year

# Aggregate new deaths by year and state
df_final = df_melted.groupby(['Province_State', 'Year'])['New_Deaths'].sum().unstack(fill_value=0)

# Add a total row
df_final.loc['Total'] = df_final.sum()

# Filter for California
california_data = df_melted[df_melted['Province_State'] == 'California'].groupby('Date')['New_Deaths'].sum()

# Clip negative values (data corrections)
california_data = california_data.clip(lower=0)

## Rolling average
rolling_avg = california_data.rolling(window=7).mean()

# Key dates to highlight
important_dates = {
    'First Vaccine (Dec 14, 2020)': '2020-12-14',
    'American Rescue Plan (Mar 11, 2021)': '2021-03-11',
    'Vaccines Open to Adults (Apr 19, 2021)': '2021-04-19'
}

# --- Plotting ---

plt.figure(figsize=(14, 7))
plt.bar(california_data.index, california_data.values, label='Daily New Deaths', color='skyblue', alpha=0.5)
plt.plot(rolling_avg.index, rolling_avg.values, label='7-Day Rolling Average', color='darkred', linewidth=3)
plt.title('California Daily New COVID-19 Deaths and 7-Day Rolling Average') # Add the title here
# ... (rest of your plotting code) ...
plt.tight_layout()

# Add vertical lines with improved label spacing
# Add vertical lines with labels offset from the line
for i, (label, date) in enumerate(important_dates.items()):
    date_obj = pd.to_datetime(date)
    plt.axvline(date_obj, color='red', linestyle='--', linewidth=1)
    
    # Adjust the horizontal and vertical position of the label for clarity
    plt.text(date_obj + pd.Timedelta(days=5),  # moved slightly right
             california_data.max() * (1 - 0.05 * i),  # staggered vertically
             label, rotation=90, color='red', fontsize=8,
             verticalalignment='top', horizontalalignment='left')  # changed alignment

# Optional: Add shaded region for rollout phase
plt.axvspan(pd.to_datetime('2020-12-14'), pd.to_datetime('2021-04-19'), 
            color='green', alpha=0.1, label='Vaccine Rollout Phase')


# Annotate the peak with adjusted position
peak_date = pd.to_datetime('2021-02-24')  # manual override instead of .idxmax()
peak_value = california_data.loc[peak_date]

plt.annotate(
    f'Peak: {int(peak_value)} deaths\n{peak_date.date()}',
    xy=(peak_date, peak_value),
    xytext=(peak_date - pd.Timedelta(days=100), peak_value - 600),  # shifted left & downward
    arrowprops=dict(arrowstyle='->', color='black'),
    fontsize=9,
    color='black',
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.7)  # optional: makes it readable
)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Print the final dataset
print(df_final)

plt.figure(figsize=(15, 10))
sns.heatmap(df_final.drop('Total'), cmap='Reds', annot=True, fmt='.0f')
plt.title('COVID-19 Deaths by State and Year 2020 - 2023')
plt.xlabel('Year')
plt.ylabel('State')
plt.tight_layout()
plt.show()











# --------------------------------------
# LASSA FEVER DATA PROCESSING (New Code)
# --------------------------------------

lassa_df = pd.read_csv('pgph.0002159.s004.csv')
def clean_lassa_data(df,year_column ='year', country_column = 'country'):
    df[year_column] = pd.to_numeric(df[year_column], errors='coerce')
    df = df.dropna(subset=[year_column])
    df = df[~df[year_column].between(2008, 2018)]

    df.drop(columns=['note', 'reference', 'status', 'source','complete_year', 'region'], inplace=True)  # Drop irrelevant columns
    

    df['confirmed_cases'] = df['confirmed_cases'].fillna(0).astype(int)  # Fill missing cases and convert to int
     # Handle missing regions
    df['country'] = df['country'].str.title().str.strip()  # Standardize country names
    df = df.groupby([year_column, country_column], as_index=False)['confirmed_cases'].sum()
    df.sort_values(by=[year_column, country_column], inplace=True)  # Optional sorting

    return df

lassa_df = clean_lassa_data(lassa_df)

print(lassa_df)

# Assuming lassa_df is your cleaned DataFrame
lassa_df['country_grouped'] = lassa_df['country'].apply(lambda x: 'Other Countries' if x != 'Nigeria' else 'Nigeria')

# Set style and figure size
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Create the line plot with the new grouped column
sns.lineplot(data=lassa_df, x='year', y='confirmed_cases', hue='country_grouped', marker='o')
# Add labels and title
plt.xlabel("Year")
plt.ylabel("Number of Confirmed Cases")
plt.title("Confirmed Lassa Fever Cases Over Time (Nigeria vs. Other Western African Countries)")
plt.legend(title='Country Group')
plt.xticks(lassa_df['year'].unique())
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

#--------------
#combination

#-----------

data = {
    'Disease': ['COVID (CA)', 'Lassa (NG)'],
    '25% Peak': [21, 48],
    '50% Peak': [39, 77],
    '75% Peak': [56, 103]
}

df_milestones = pd.DataFrame(data)

# Set index for stacked bar
df_milestones.set_index('Disease', inplace=True)

# Stacked bar chart
df_milestones.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='coolwarm')
plt.title('Days from First Confrimed Death to Peak Number of Deaths Milestones (California & Nigeria) (25%, 50%, 75% (of total the deaths))')
plt.ylabel('Days')
plt.xlabel('Disease and Location')
plt.legend(title='Peak Milestone')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


