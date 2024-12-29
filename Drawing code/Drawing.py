from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump,load
import gc
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Function to generate a 3D bar plot for classifier metrics (e.g., ACC, F1, G-mean, RANK)
def plot_classifier_metrics(file_path, classifier_field, regex_pattern, output_file_path):
    """
    Generates a 3D bar plot for the selected classifier metric (ACC, F1, GM, RANK).

    Parameters:
    - file_path (str): Path to the input Excel file.
    - classifier_field (str): Field to extract (e.g., 'ACC', 'F1', 'GM', 'RANK').
    - regex_pattern (str): Regular expression pattern to extract the specific metric.
    - output_file_path (str): Path to save the output image.
    """
    # Load the Excel file
    data_new = pd.ExcelFile(file_path)

    # Function to extract data based on classifier field and regex pattern
    def extract_classifier_data(sheet_name, classifier_field, regex_pattern):
        sheet_data = data_new.parse(sheet_name)
        
        # Extract columns related to the specified classifier field using regex
        classifier_columns = [col for col in sheet_data.columns if classifier_field in col]
        
        if not classifier_columns:
            return pd.Series([])  # Return empty series if no matching columns found
        
        # Extract data based on the provided regex pattern
        classifier_data = sheet_data[classifier_columns[0]]  # Use the first matching column
        extracted_values = classifier_data.apply(
            lambda x: float(re.search(regex_pattern, x).group(1)) if isinstance(x, str) else None
        )
        return extracted_values

    # Collect classifier data (e.g., F1, G-mean, etc.) from all sheets
    classifier_data_combined = {}
    for sheet in data_new.sheet_names:
        classifier_data_combined[sheet] = extract_classifier_data(sheet, classifier_field, regex_pattern)

    # Combine the data into a DataFrame for plotting
    classifier_data_df = pd.DataFrame(classifier_data_combined)
    classifier_data_df['Method'] = data_new.parse(data_new.sheet_names[0])['BR1:400']
    classifier_data_df.set_index('Method', inplace=True)

    classifier_data_df.dropna(inplace=True)  # Drop rows with missing data

    # Prepare data for 3D bar plot across multiple sheets
    x = np.arange(len(classifier_data_df.index))  # X-axis positions for methods
    y = np.arange(len(classifier_data_df.columns))  # Y-axis positions for sheets
    x_grid, y_grid = np.meshgrid(x, y)  # Create grid for X and Y
    x_pos = x_grid.flatten()
    y_pos = y_grid.flatten()
    z_pos = np.zeros_like(x_pos)  # Z-axis base positions

    dx = dy = 0.8  # Width and depth of bars
    dz = classifier_data_df.values.T.flatten()  # Heights from classifier data values

    # Create a colormap for the bar colors (use a gradient)
    colors = cm.viridis((dz - np.min(dz)) / (np.max(dz) - np.min(dz)))  # Normalize to colormap range

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw bars with color gradient
    bars = ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, alpha=0.7)

    # Add text labels for each bar (extracted value)
    for i in range(len(x_pos)):
        ax.text(x_pos[i], y_pos[i], dz[i] + 0.01, f'{dz[i]:.4f}', color='black', ha='center', va='bottom', fontsize=10)

    # Customize the axes
    ax.set_xticks(np.arange(len(classifier_data_df.index)))
    ax.set_xticklabels(classifier_data_df.index, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(classifier_data_df.columns)))
    ax.set_yticklabels(classifier_data_df.columns)
    ax.set_zlabel(f'{classifier_field} Value')
    ax.set_title(f'{classifier_field} Value Across Different BR')

    # Save the figure as an image
    plt.tight_layout()
    plt.savefig(output_file_path)

    # Show the plot
    plt.show()

    # Return the output file path
    return output_file_path
    
"""
# Example Usage
file_path = 'D:/jupyternotebook/X-1/CWT-IL-ACSAWGAN-GP/文本文件/CWRU/BR汇总.xlsx'
classifier_field = 'VGG-16'  # For VGG-16
regex_pattern_acc = r'(\d+\.\d+)%/.*'  # Regex to extract ACC value (e.g., '46.80%')
regex_pattern_f1 = r'^.*?/(\d+\.\d+)\/.*$'   # Regex to extract F1 value (e.g., '0.4740')
regex_pattern_gm = r'.*/(\d+\.\d+)/.*' # Regex to extract G-mean value (e.g., '0.6636')
regex_pattern_rank = r'.*/(\d+)$'      # Regex to extract RANK value (e.g., '8')

# Call the function to generate the plot for ACC
output_file_path_acc = 'vgg16_acc_comparison.png'
plot_classifier_metrics(file_path, 'VGG-16', regex_pattern_acc, output_file_path_acc)

# Call the function to generate the plot for F1
output_file_path_f1 = 'vgg16_f1_comparison.png'
plot_classifier_metrics(file_path, 'VGG-16', regex_pattern_f1, output_file_path_f1)

# Call the function to generate the plot for G-mean
output_file_path_gm = 'vgg16_gm_comparison.png'
plot_classifier_metrics(file_path, 'VGG-16', regex_pattern_gm, output_file_path_gm)

# Call the function to generate the plot for RANK
output_file_path_rank = 'vgg16_rank_comparison.png'
plot_classifier_metrics(file_path, 'VGG-16', regex_pattern_rank, output_file_path_rank)
"""

def plot_max_min_values_with_shading_v5(max_values_df, min_values_df, metric, save_path=None):
    """
    Generate a line plot for the max and min values with shading between the curves.
    The plot will also include grid lines in the background.
    Each model's max and min curves will have the same color, and different models will use different colors.
    The legend will display a color block for each model, representing the corresponding max and min curves.
    
    Parameters:
    - max_values_df: DataFrame containing the maximum values for each method and classifier.
    - min_values_df: DataFrame containing the minimum values for each method and classifier.
    - save_path: Optional path to save the plot as an image. If None, the plot will be shown but not saved.
    """
    # Ensure all values are numeric
    max_values_df = max_values_df.apply(pd.to_numeric, errors='coerce')
    min_values_df = min_values_df.apply(pd.to_numeric, errors='coerce')

    # Generate a list of colors for the plots. Use 'tab10' colormap, which provides 10 distinct colors
    colors = plt.cm.get_cmap('tab10', len(max_values_df.index))

    # Plotting
    plt.figure(figsize=(12, 8))  # Set the size of the plot

    # List to store the custom legend handles
    legend_handles = []

    # Iterate over each row (method) and plot the max and min values with shading between them
    for idx, color in zip(max_values_df.index, colors.colors):
        x = np.array(max_values_df.columns)  # BR values (x-axis)
        max_y = np.array(max_values_df.loc[idx])  # Max values for this method (y-axis)
        min_y = np.array(min_values_df.loc[idx])  # Min values for this method (y-axis)

        # Plot max and min values as lines (without points) with the same color for both
        plt.plot(x, max_y, color=color, linewidth=2)
        plt.plot(x, min_y, color=color, linewidth=2)

        # Fill the area between the max and min values with shading
        plt.fill_between(x, min_y, max_y, color=color, alpha=0.3)

        # Add a custom legend handle with a color block for each model
        legend_handles.append(Line2D([0], [0], color=color, lw=6, label=idx))

    # Customize plot labels and title
    plt.title(f'{metric} range for Each Method', fontsize=16)
    plt.xlabel('BR Values', fontsize=12)
    plt.ylabel('Value', fontsize=12)

    # Add gridlines to the background with a slight gray color for better contrast
    plt.grid(True, linestyle='--', alpha=0.7, color='gray')

    # Create a custom legend with color blocks for each model
    plt.legend(handles=legend_handles, title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to make room for legend
    plt.tight_layout()

    # If save_path is provided, save the plot as an image
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # Show the plot
    plt.show()

def plot_composite_bars(df1, df2, metric,bar_width=0.1,save_path=None):
    """
    Draw a composite bar chart comparing the bars under different BRs in two data frames.
    The values ​​under each BR are displayed in different colors, with smaller values ​​as the base and larger values ​​as the additive.

    Parameters:
    df1 (DataFrame): Contains the data of the first data frame, with the index as the method name and the columns as the BR values.
    df2 (DataFrame): Contains the data of the second data frame, with the index as the method name and the columns as the BR values.
    bar_width (float): The width of the bar chart.
    """
    br_columns = df1.columns

    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors1 = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    colors2 = ['tab:cyan', 'tab:olive', 'tab:gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    index = np.arange(len(df1))  
    

    legend_handles = []
    legend_labels = []
    

    for col_idx, col in enumerate(br_columns):
        color1 = colors1[col_idx]  
        color2 = colors2[col_idx]  

        legend_handles.append(plt.Line2D([0], [0], color=color1, lw=4))
        legend_labels.append(f'{col} - CWRU')
        
        legend_handles.append(plt.Line2D([0], [0], color=color2, lw=4))
        legend_labels.append(f'{col} - SEU')

        for i in range(len(df1)):
            val1 = df1.iloc[i, col_idx]
            val2 = df2.iloc[i, col_idx]
            
            larger_value = max(val1, val2)
            smaller_value = min(val1, val2)
            
            ax.barh(index[i] + col_idx * 0.1, smaller_value, height=bar_width, color=color1, edgecolor='black')
            
            ax.barh(index[i] + col_idx * 0.1, larger_value - smaller_value, height=bar_width, color=color2, edgecolor='black', left=smaller_value)
    
    ax.set_xlabel('Values', fontsize=12)
    ax.set_ylabel('Models', fontsize=12)
    ax.set_title(f'Composite bar chart comparison of classifier {metric} mean at different BRs', fontsize=14)
    
    ax.set_yticks(index + 0.1 * (len(br_columns) - 1) / 2)  
    ax.set_yticklabels(df1.index)  
    
    ax.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  
        print(f"Save to {save_path}")
    
    plt.show()

# plot_composite_bars(acc_mean_values_df_cwru, acc_mean_values_df_seu,metric='ACC',save_path='acc_composite_bars_plot.png')