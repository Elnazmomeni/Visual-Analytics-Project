import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('normalized_numeric_data.csv')
data_artist = pd.read_csv('cleaned_dataset_tracks.csv')

# Merge datasets for song selection
full_data = pd.merge(data_artist[['name', 'artists']], data, left_index=True, right_index=True, how="inner")

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"])
app.title = "Interactive Visualization Dashboard"

# Layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("Interactive Visualization Dashboard", className="text-center mb-4"),
        html.Div([
            html.Label("Select Dimensionality Reduction Method:"),
            dcc.RadioItems(
                id='method',
                options=[
                    {'label': 'PCA', 'value': 'PCA'},
                    {'label': 't-SNE', 'value': 'TSNE'}
                ],
                value='PCA',
                inline=True
            )
        ], className="mb-3"),

        html.Div(id="slider-container", children=[
            html.Label("Number of Components:"),
            dcc.Slider(
                id='n_components',
                min=2, max=10, step=1, value=2,
                marks={i: str(i) for i in range(2, 11)}
            )
        ], className="mb-3"),

        html.Div(id="tsne-container", children=[
            html.Label("t-SNE Perplexity (only for t-SNE):"),
            dcc.Slider(
                id='perplexity',
                min=5, max=50, step=5, value=20,
                marks={i: str(i) for i in range(5, 51, 5)},
                tooltip={"placement": "bottom"}
            )
        ], className="mb-3"),

        dcc.Store(id='sampled-data-store'),  # Store for sharing sampled data
        dcc.Graph(id='scatter_plot', style={'height': '70vh'}),

       html.Div(id='analysis-results', className="mt-4"),
        html.Div(id='song-analysis', className='mt-4'),
        
        html.Hr(),
        html.H3("Song Recommendation System", className="text-center mt-4"),
        
        html.Div([
            html.Label("Search for a Song:"),
            dcc.Input(id='search-song', type='text', placeholder='Enter song name!', debounce=True, className="form-control mb-2"),
            dcc.Dropdown(id='song-dropdown', placeholder='Select a song', className="mb-3"),
            html.Button("Get Recommendations", id='recommend-btn', n_clicks=0, className="btn btn-primary"),
        ], className="mb-4"),

        html.Div(id='recommendation-results', className="mt-4"),
    ], className="container")
])

# Callback to toggle the slider visibility
@app.callback(
    [Output('slider-container', 'style'),
     Output('tsne-container', 'style')],
    Input('method', 'value')
)
def toggle_slider_visibility(method):
    if method == 'PCA':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

# Combined callback to update scatter plot and store data
@app.callback(
    [Output('scatter_plot', 'figure'), Output('sampled-data-store', 'data')],
    [Input('method', 'value'),
     Input('n_components', 'value'),
     Input('perplexity', 'value')]
)
def update_plot_and_store_data(method, n_components, perplexity):
    if method == 'PCA':
        pca = PCA(n_components=n_components)
        result = pca.fit_transform(data)
        explained_variance = sum(pca.explained_variance_ratio_) * 100
        title = f"PCA Scatter Plot (Explained Variance: {explained_variance:.2f}%)"
        
        result_df = pd.DataFrame(result, columns=[f"Component {i+1}" for i in range(n_components)])
        result_df['Popularity'] = data['popularity']
        fig = px.scatter(
            result_df,
            x='Component 1', y='Component 2',
            color='Popularity',
            color_continuous_scale='Viridis',
            title=title
        )
        fig.update_layout(clickmode='event+select', dragmode='lasso')
        return fig, data.to_dict('records')

    elif method == 'TSNE':
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        sampled_data = data.sample(10000, random_state=42).copy()
        tsne_result = tsne.fit_transform(sampled_data)
        sampled_data['Component 1'] = tsne_result[:, 0]
        sampled_data['Component 2'] = tsne_result[:, 1]
        fig = px.scatter(
            sampled_data,
            x='Component 1', y='Component 2',
            color='popularity',
            color_continuous_scale='Plasma',
            title="t-SNE Scatter Plot"
        )
        fig.update_layout(clickmode='event+select', dragmode='lasso')
        return fig, sampled_data.to_dict('records')

# Callback to analyze selected data
@app.callback(
    Output('analysis-results', 'children'),
    [Input('scatter_plot', 'selectedData')],
    [State('sampled-data-store', 'data')]
)
def analyze_selected_data(selected_data, stored_data):
    if not selected_data or 'points' not in selected_data:
        return html.Div("Select points on the plot to see the analysis.", className="text-dark text-left mt-4")
    
    if stored_data is None:
        return html.Div("No data available for analysis.", className="text-danger")
    
    sampled_data = pd.DataFrame(stored_data)  # Convert back to DataFrame
    selected_points = selected_data['points']
    selected_indices = [p['pointIndex'] for p in selected_points]
    selected_rows = sampled_data.iloc[selected_indices]

    # Case when only one point is selected
    if len(selected_indices) == 1:
        avg_popularity = selected_rows['popularity'].values[0]
        return html.Div([
            html.H5("Analysis of Selected Points", className="font-weight-bold text-left"),
            html.P(f"Number of Points Selected: {len(selected_indices)}", className="font-weight-bold"),
            html.P(f"Average Popularity: {avg_popularity:.2f}", className="font-weight-bold text-success", style={'fontSize': '20px'}),
            html.P("Correlation matrix cannot be calculated with just one point.", className="text-danger font-weight-bold"),
        ], className="text-dark text-left")

    # Case when multiple points are selected
    avg_popularity = selected_rows['popularity'].mean()

    # Focus on correlations with "popularity"
    correlations = selected_rows.corr()['popularity'].drop('popularity')
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    correlation_table = html.Table([
        html.Thead(
            html.Tr([html.Th("Feature"), html.Th("Correlation with Popularity")])
        ),
        html.Tbody([
            html.Tr([html.Td(feature), html.Td(f"{correlations[feature]:.2f}")]) 
            for feature in top_correlations.index
        ])
    ], className="table table-striped table-bordered")

    return html.Div([
        html.H5("Analysis of Selected Points", className="font-weight-bold text-left"),
        html.P(f"Number of Points Selected: {len(selected_indices)}", className="font-weight-bold"),
        html.P(f"Average Popularity: {avg_popularity:.2f}", className="font-weight-bold text-primary", style={'fontSize': '20px'}),
        html.H6("Top Correlations with Popularity:", className="font-weight-bold text-left mt-3"),
        correlation_table,
    ], className="text-dark text-left")

# Callback for song analysis
@app.callback(
    Output('song-analysis', 'children'),
    [Input('scatter_plot', 'selectedData')],
    [State('sampled-data-store', 'data')]
)
def song_analysis(selected_data, stored_data): 
    if not selected_data or 'points' not in selected_data:
        return html.Div("Select points on the plot to analyze.", className="text-left text-dark")
    
    if stored_data is None:
        return html.Div("No data available for analysis.", className="text-danger")
    
    sampled_data = pd.DataFrame(stored_data)  # Convert back to DataFrame
    selected_indices = [p['pointIndex'] for p in selected_data['points']]
    selected_rows = sampled_data.iloc[selected_indices]  # Get selected rows

    # Check if the 'artists' column exists and handle gracefully if it doesn't
    if 'artists' not in data_artist.columns:
        return html.Div("The 'artists' column is not available in the dataset.", className="text-danger")

    # Extract top 10 artists from the selected points
    selected_rows = pd.merge(
        selected_rows,
        data_artist[['artists', 'name', 'release_date']],
        left_index=True,
        right_index=True,
        how='left'
    )
    
    # If there are no artists in the selected points
    if selected_rows['artists'].isnull().all():
        return html.Div("No artist data found for the selected points.", className="text-warning")

    # Get the top 10 artists
    top_artists = selected_rows['artists'].value_counts().head(10)

    # Get the names and release dates of songs from these artists
    top_songs = selected_rows[selected_rows['artists'].isin(top_artists.index)][['name', 'release_date', 'artists']].head(10)

    # Create a table to show the top songs and their release dates
    song_table = html.Table([
        html.Thead(
            html.Tr([html.Th("Song Name"), html.Th("Release Date"), html.Th("Artist")])
        ),
        html.Tbody([
            html.Tr([html.Td(row['name']), html.Td(row['release_date']), html.Td(row['artists'])])
            for _, row in top_songs.iterrows()
        ])
    ], className="table table-striped table-bordered")

    return html.Div([
        html.H5("Analysis of Selected Songs", className="font-weight-bold text-left"),
        html.P(f"Number of Points Selected: {len(selected_indices)}", className="font-weight-bold"),

        # Display top artists
        html.H6("Top Artists:", className="font-weight-bold text-left mt-3"),
        html.Ul([html.Li(f"{artist} ({count} songs)") for artist, count in top_artists.items()]),

        # Display Song Data
        html.H6("Top Songs from Selected Artists:", className="font-weight-bold text-left mt-3"),
        song_table,
    ], className="text-dark text-left")
    
    
    # Callback for song search
@app.callback(
    Output("song-dropdown", "options"),
    Input("search-song", "value")
)
def update_song_dropdown(search_value):
    if not search_value:
        return []
    matches = full_data[full_data["name"].str.contains(search_value, case=False, na=False)]
    return [{"label": f"{row['name']} - {row['artists']}", "value": row["name"]} for _, row in matches.iterrows()]

# Callback for music recommendation
@app.callback(
    Output("recommendation-results", "children"),
    Input("recommend-btn", "n_clicks"),
    State("song-dropdown", "value")
)
def recommend_songs(n_clicks, selected_song):
    if not selected_song:
        return html.Div("Select a song to get recommendations.", className="text-warning")

    selected_song_idx = full_data[full_data["name"] == selected_song].index
    if selected_song_idx.empty:
        return html.Div("Song not found in dataset.", className="text-danger")
    
    selected_song_idx = selected_song_idx[0]
    selected_song_features = data.iloc[selected_song_idx].values.reshape(1, -1)

    # Use Nearest Neighbors for recommendations
    nn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
    nn_model.fit(data)
    distances, indices = nn_model.kneighbors(selected_song_features)

    recommended_songs_idx = indices.flatten()[1:]  # Exclude the first (itself)
    recommended_songs = full_data.iloc[recommended_songs_idx][['name', 'artists']]

    return html.Div([
        html.H5("Recommended Songs", className="font-weight-bold"),
        html.Ul([html.Li(f"{row['name']} - {row['artists']}") for _, row in recommended_songs.iterrows()])
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
