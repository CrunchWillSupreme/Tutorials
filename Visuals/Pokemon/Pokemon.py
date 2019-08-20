# https://elitedatascience.com/python-seaborn-tutorial

# Pandas for managing datasets
import pandas as pd
# Matplotlib for additional customization
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
# Seaborn for plotting and styling
import seaborn as sns

# Read dataset
df = pd.read_csv(r'C:\Codes\Seaborn_Pokemon\Pokemon.csv', index_col = 0, encoding = "ISO-8859-1")

# Display first 5 observations
df.head()

# Default Scatter Plot
# Recommended way
sns.lmplot(x = 'Attack', y = 'Defense', data = df)

# Scatterplot parameters
sns.lmplot(x = 'Attack', y = 'Defense', data = df,
           fit_reg = False,# No regression line
          hue = 'Stage') # Color by evolution stage
          
# Plot using Seaborn
sns.lmplot(x = 'Attack', y = 'Defense', data = df,
          fit_reg = False,
          hue = 'Stage')

# Tweak using Matplotlib
plt.ylim(0, None)
plt.xlim(0, None)

# Default boxplot
sns.boxplot(data = df)

# Pre-format DataFrame
stats_df = df.drop(['Total', 'Stage', 'Legendary'], axis = 1)
# New boxplot using stats_df
sns.boxplot(data = stats_df)

# Set theme
sns.set_style('whitegrid')

# Violin plot
sns.violinplot(x = 'Type 1', y = 'Attack', data = df)

# Pokemon color palette
pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]
                   
# Violin plot with Pokemon color palette
sns.violinplot(x = 'Type 1', y = 'Attack', data = df,
              palette = pkmn_type_colors)
              
# Swarm plot with Pokemon color palette
sns.swarmplot(x = 'Type 1', y = 'Attack', data = df,
             palette = pkmn_type_colors)
             
# ###### Overlaying plots

# 1. Make our figure larger using Matplotlib.
# 2. Plot the violin plot.  However, set 'inner = None' to remove the bars inside the violins.
# 3. Plot the swarm plot.  Make points black so they pop out more.
# 4. Finally, set a title using Matplotlib.

# Set figure size with matplotlib
plt.figure(figsize = (10,6))

# Create plot
sns.violinplot(x = 'Type 1', y = 'Attack', data = df,
              inner = None, # Remove the bars inside the violin plot
              palette = pkmn_type_colors)

sns.swarmplot(x = 'Type 1', y = 'Attack', data = df,
             color = 'k', # Make points black
             alpha = 0.7) # and slightly transparent

# Set title with matplotlib
plt.title('Attack by Type')

# All of our stats are in separate columns.  Instead we want to "melt" them into one column.
# 
# To do so, we'll use Panda's melt() function.  It takes 3 arguments:
# 
# 1. Dataframe to melt.
# 2. ID variables to keep (Pandas will melt all of the other ones).
# 3. Finally, a name for the new, melted variable.

# Melt Dataframe
melted_df = pd.melt(stats_df,
                   id_vars = ['Name', 'Type 1', 'Type 2'], # Variables to keep                  
                    var_name = 'Stat') # Name of melted variable
melted_df.head()

# Let's make a swarm plot with melted_df.
# 
# - This time we're going to set x = 'Stat' and y = 'value' so our swarms are separated by stat.
# - Then, we'll set hue = 'Type 1' to color our points by the Pokemon type.

# Swarmplot with melted_df
sns.swarmplot(x = 'Stat', y = 'value', data = melted_df,
             hue = 'Type 1')
             
# Let's make a few final tweaks for a more readable chart:
# 
# 1. Enlarge the plot.
# 2. Separate points by hue using the argument split = True.
# 3. Use our custom Pokemon color palette.
# 4. Adjust the y-axis limits to end at 0.
# 5. Place the legend to the right.

#Customizations
# 1. Enlarge the plot
plt.figure(figsize = (10,6))

sns.swarmplot(x = 'Stat', y = 'value', data = melted_df,
             hue = 'Type 1', dodge = True, # 2. Separate points by hue
             palette = pkmn_type_colors) # 3. Use Pokemon palette

# 4. Adjust the y-axis
plt.ylim(0, 260)

# 5. Place legend to the right
plt.legend(bbox_to_anchor=(1, 1), loc = 2)

# Heatmap
# Calculate correlations
corr = stats_df.corr()

sns.heatmap(corr)

# Histogram
sns.distplot(df.Attack)

# Barplot
sns.countplot(x = 'Type 1', data = df, palette=pkmn_type_colors)

# Rotate x-labels
plt.xticks(rotation=-45)

# Factor plot- make it easy to separate plots by categorical classes.
g = sns.factorplot(x = 'Type 1',
                  y = 'Attack',
                  data = df,
                  hue = 'Stage', # Color by stage
                  col = 'Stage', # Separate by stage
                  kind = 'swarm') # Swarmplot

# Rotate x-axis labels
g.set_xticklabels(rotation=-45)

# Density Plot - Displays the distribution between two variables
sns.kdeplot(df.Attack, df.Defense)

# Joint Distribution Plot
sns.jointplot(x = 'Attack', y = 'Defense', data = df)



