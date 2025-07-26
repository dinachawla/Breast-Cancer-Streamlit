import plotly.express as px
import pandas as pd

# Get user inputs in order
ordered_keys = [f[0] for group in FEATURE_GROUPS.values() for f in group]
user_input = np.array([values[k] for k in ordered_keys])

# Calculate average profiles
benign_profile = df[df['target'] == 1][ordered_keys].mean()
malignant_profile = df[df['target'] == 0][ordered_keys].mean()

# Normalize all
scaler = MinMaxScaler()
scaled = scaler.fit_transform([benign_profile, malignant_profile, user_input])
benign_scaled, malignant_scaled, user_scaled = scaled

# Build comparison DataFrame
profile_df = pd.DataFrame({
    'Feature': ordered_keys * 3,
    'Value': np.concatenate([benign_scaled, malignant_scaled, user_scaled]),
    'Profile': ['Benign'] * len(ordered_keys) +
               ['Malignant'] * len(ordered_keys) +
               ['Your Input'] * len(ordered_keys)
})

# Generate bar chart
fig = px.bar(
    profile_df,
    x='Feature',
    y='Value',
    color='Profile',
    barmode='group',
    title='Live Tumor Feature Comparison (Normalized)',
    height=500
)
fig.update_layout(xaxis_tickangle=-45)

st.plotly_chart(fig, use_container_width=True)
