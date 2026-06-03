import pandas as pd

# Load the CSV file
df = pd.read_csv("keypoint.csv")

# Define a mapping function
def modify_first_column(value):
    if value == 5:
        return 4
    # elif value in [2, 3]:
    #     return 1
    # elif value in [4, 5]:
    #     return 2
    else:
        return value  # Keep other values unchanged

# Apply the function to the first column
df.iloc[:, 0] = df.iloc[:, 0].apply(modify_first_column)

# Save the modified CSV
df.to_csv("modified_keypoints2.csv", index=False)

# Display the modified dataframe
import ace_tools as tools
tools.display_dataframe_to_user(name="Modified Data", dataframe=df)
