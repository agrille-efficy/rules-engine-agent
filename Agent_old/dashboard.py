import json
import pandas as pd
from dash import Dash, html, dcc, Input, Output, dash_table, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from datetime import datetime
import os
import glob
from pathlib import Path

# Initialize the Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Data Mapping Visualization Dashboard"
app.config.suppress_callback_exceptions = True  # Fix for dynamic components

class MappingDashboard:
    def __init__(self, results_dir=None):
        self.results_dir = results_dir or r"c:\Users\axel.grille\Documents\rules-engine-agent\Agent\RAG\results_for_agent"
        self.sample_data = {
            "mapping_visualization": {
                "source_structure": {
                    "file_name": "Mails.csv",
                    "file_type": "CSV",
                    "total_rows": 5,
                    "total_columns": 25,
                    "columns": [
                        "VISITE_REF", "CL_REF", "Société - Nom", "Objet", "Date saisie",
                        "Date", "Type", "Canal", "Catégorie", "Auteur", "Dossier",
                        "Mail expéditeur", "Corps du message", "Programmé pour",
                        "Mail destinataire", "VISITE_MAIL_CC", "Etat", "Date de suivi",
                        "État de la Catégorie", "Date fin", "Lieu", "Désignation",
                        "Référence campagne", "Diffusion", "Priorité"
                    ],
                    "domain_detected": "communication and messaging"
                },
                "target_structure": {
                    "table_name": "Mail",
                    "table_code": "mail",
                    "selection_reason": "The 'Mail' table is an entity type table designed to manage and track email interactions, which aligns well with the domain of the source CSV.",
                    "confidence_score": 0.85
                },
                "mapping_details": {
                    "successful_mappings": 9,
                    "unmapped_fields_count": 5,
                    "field_mappings": [
                        {"source_field": "VISITE_REF", "target_field": "mailInteractionKey", "data_type": "INTEGER", "transformation": "none", "confidence": 0.9},
                        {"source_field": "CL_REF", "target_field": "mailInteractionKey", "data_type": "INTEGER", "transformation": "none", "confidence": 0.85},
                        {"source_field": "Société - Nom", "target_field": "mailSysCreatedUserKey", "data_type": "VARCHAR(16)", "transformation": "uppercase", "confidence": 0.88},
                        {"source_field": "Objet", "target_field": "mailSubject", "data_type": "VARCHAR(67)", "transformation": "none", "confidence": 0.92},
                        {"source_field": "Date saisie", "target_field": "mailSysCreatedDate", "data_type": "DATE", "transformation": "date_format", "confidence": 0.87},
                        {"source_field": "Mail expéditeur", "target_field": "mailSysCreatedUserKey", "data_type": "VARCHAR(22)", "transformation": "none", "confidence": 0.89},
                        {"source_field": "Corps du message", "target_field": "mailBody", "data_type": "TEXT", "transformation": "none", "confidence": 0.93},
                        {"source_field": "Mail destinataire", "target_field": "mailRecipient", "data_type": "VARCHAR(34)", "transformation": "none", "confidence": 0.9},
                        {"source_field": "Etat", "target_field": "mailSentStatus", "data_type": "VARCHAR(9)", "transformation": "uppercase", "confidence": 0.88}
                    ],
                    "unmapped_fields": [
                        {"source_field": "Dossier", "reason": "no suitable target found"},
                        {"source_field": "Date de suivi", "reason": "no suitable target found"},
                        {"source_field": "État de la Catégorie", "reason": "no suitable target found"},
                        {"source_field": "Lieu", "reason": "no suitable target found"},
                        {"source_field": "Référence campagne", "reason": "no suitable target found"}
                    ]
                },
                "summary": {
                    "mapping_success_rate": 36.0,
                    "ingestion_strategy": "requires_transformation",
                    "estimated_success_rate": 0.88,
                    "requires_transformation": True,
                    "ready_for_ingestion": False
                }
            }
        }
        
    def get_available_analysis_files(self):
        """Get all available analysis files from the results directory"""
        try:
            pattern = os.path.join(self.results_dir, "*_ingestion_analysis.json")
            files = glob.glob(pattern)
            
            # Get file info with timestamps
            file_info = []
            for file_path in files:
                stat = os.stat(file_path)
                filename = os.path.basename(file_path)
                source_file = filename.replace("_ingestion_analysis.json", "")
                
                file_info.append({
                    'path': file_path,
                    'filename': filename,
                    'source_file': source_file,
                    'modified_time': datetime.fromtimestamp(stat.st_mtime),
                    'size': stat.st_size
                })
            
            # Sort by modification time (newest first)
            file_info.sort(key=lambda x: x['modified_time'], reverse=True)
            return file_info
            
        except Exception as e:
            print(f"Error getting analysis files: {e}")
            return []
    
    def get_latest_analysis_file(self):
        """Get the most recently modified analysis file"""
        files = self.get_available_analysis_files()
        return files[0]['path'] if files else None
    
    def create_file_selector_dropdown(self):
        """Create dropdown for selecting analysis files"""
        files = self.get_available_analysis_files()
        
        if not files:
            return html.Div([
                dbc.Alert("No analysis files found. Run the agent on some data files first!", color="warning")
            ])
        
        options = []
        for file_info in files:
            label = f"{file_info['source_file']} ({file_info['modified_time'].strftime('%Y-%m-%d %H:%M')})"
            options.append({'label': label, 'value': file_info['path']})
        
        return dbc.Row([
            dbc.Col([
                html.Label("Select Analysis to Visualize:", className="fw-bold"),
                dcc.Dropdown(
                    id='file-selector',
                    options=options,
                    value=files[0]['path'],  # Default to latest
                    placeholder="Choose an analysis file...",
                    className="mb-3"
                )
            ], width=8),
            dbc.Col([
                dbc.Button(
                    "🔄 Refresh Files", 
                    id="refresh-button", 
                    color="primary", 
                    size="sm",
                    className="mt-4"
                )
            ], width=4)
        ])
    
    def load_mapping_data_from_file(self, analysis_file_path):
        """Load and convert analysis data to mapping visualization format"""
        try:
            with open(analysis_file_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            # Convert the analysis data to mapping visualization format
            # This is a conversion function for your existing analysis files
            mapping_viz = self.convert_analysis_to_mapping_viz(analysis_data)
            return {"mapping_visualization": mapping_viz}
        except Exception as e:
            print(f"Error loading file {analysis_file_path}: {e}")
            return self.sample_data
    
    def convert_analysis_to_mapping_viz(self, analysis_data):
        """Convert analysis data format to mapping visualization format"""
        # Extract basic file structure
        file_structure = analysis_data.get("file_structure", {})
        recommended = analysis_data.get("recommended_ingestion", {})
        table_options = analysis_data.get("table_options", [])
        
        # Get the primary recommended table
        primary_table = table_options[0] if table_options else {}
        
        # Create mock mapping data based on the structure
        # In a real scenario, you'd have actual field mappings
        source_columns = file_structure.get("columns", [])
        mock_mappings = []
        
        # Create some sample mappings for demonstration
        # You can replace this with your actual mapping logic
        target_fields = ["mailKey", "mailSubject", "mailBody", "mailSysCreatedDate", 
                        "mailRecipient", "mailSender", "mailStatus", "mailType", "mailCategory"]
        
        for i, col in enumerate(source_columns[:len(target_fields)]):
            confidence = 0.75 + (i % 4) * 0.05  # Mock confidence scores
            transformation = "none" if i % 3 == 0 else ("uppercase" if i % 3 == 1 else "date_format")
            
            mock_mappings.append({
                "source_field": col,
                "target_field": target_fields[i % len(target_fields)],
                "data_type": "VARCHAR(50)" if i % 2 == 0 else "INTEGER",
                "transformation": transformation,
                "confidence": confidence
            })
        
        # Create unmapped fields
        unmapped_fields = []
        for col in source_columns[len(target_fields):]:
            unmapped_fields.append({
                "source_field": col,
                "reason": "no suitable target found"
            })
        
        return {
            "source_structure": {
                "file_name": analysis_data.get("source_file", "Unknown"),
                "file_type": "CSV",
                "total_rows": file_structure.get("total_rows", 0),
                "total_columns": len(source_columns),
                "columns": source_columns,
                "domain_detected": "auto-detected from analysis"
            },
            "target_structure": {
                "table_name": primary_table.get("table_name", "Unknown"),
                "table_code": primary_table.get("table_code", "unknown"),
                "selection_reason": f"Selected based on composite score of {primary_table.get('composite_score', 0):.3f}",
                "confidence_score": min(primary_table.get("composite_score", 1.0) / 3.0, 0.95) if primary_table.get("composite_score") else 0.5
            },
            "mapping_details": {
                "successful_mappings": len(mock_mappings),
                "unmapped_fields_count": len(unmapped_fields),
                "field_mappings": mock_mappings,
                "unmapped_fields": unmapped_fields
            },
            "summary": {
                "mapping_success_rate": (len(mock_mappings) / len(source_columns) * 100) if source_columns else 0,
                "ingestion_strategy": "requires_transformation",
                "estimated_success_rate": 0.85,
                "requires_transformation": any(m["transformation"] != "none" for m in mock_mappings),
                "ready_for_ingestion": recommended.get("ready_for_sql", False)
            }
        }

    def load_mapping_data(self, file_path=None):
        """Load mapping data from JSON file or use sample data"""
        if file_path and os.path.exists(file_path):
            return self.load_mapping_data_from_file(file_path)
        return self.sample_data
    
    def create_kpi_cards(self, data):
        """Create KPI cards for the executive summary"""
        mapping_viz = data.get("mapping_visualization", {})
        summary = mapping_viz.get("summary", {})
        mapping_details = mapping_viz.get("mapping_details", {})
        
        success_rate = summary.get("mapping_success_rate", 0)
        confidence = mapping_viz.get("target_structure", {}).get("confidence_score", 0)
        mapped_fields = mapping_details.get("successful_mappings", 0)
        total_fields = len(mapping_viz.get("source_structure", {}).get("columns", []))
        
        # Define color based on success rate
        def get_color(value, thresholds=[50, 75]):
            if value >= thresholds[1]: return "success"
            elif value >= thresholds[0]: return "warning" 
            else: return "danger"
        
        cards = [
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{success_rate:.1f}%", className="card-title text-center"),
                    html.P("Mapping Success Rate", className="card-text text-center")
                ])
            ], color=get_color(success_rate), outline=True, className="mb-3"),
            
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{confidence:.0%}", className="card-title text-center"),
                    html.P("Target Confidence", className="card-text text-center")
                ])
            ], color=get_color(confidence*100), outline=True, className="mb-3"),
            
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{mapped_fields}/{total_fields}", className="card-title text-center"),
                    html.P("Fields Mapped", className="card-text text-center")
                ])
            ], color="info", outline=True, className="mb-3"),
            
            dbc.Card([
                dbc.CardBody([
                    html.H4("🔄" if summary.get("requires_transformation") else "✅", className="card-title text-center"),
                    html.P("Transformation", className="card-text text-center")
                ])
            ], color="warning" if summary.get("requires_transformation") else "success", outline=True, className="mb-3")
        ]
        
        return dbc.Row([dbc.Col(card, width=3) for card in cards])
    
    def create_sankey_diagram(self, data):
        """Create Sankey diagram for field mapping visualization"""
        mapping_viz = data.get("mapping_visualization", {})
        field_mappings = mapping_viz.get("mapping_details", {}).get("field_mappings", [])
        
        if not field_mappings:
            return go.Figure().add_annotation(text="No mapping data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for Sankey
        source_fields = list(set([mapping["source_field"] for mapping in field_mappings]))
        target_fields = list(set([mapping["target_field"] for mapping in field_mappings]))
        
        # Create node labels and indices
        all_nodes = source_fields + target_fields
        node_indices = {node: i for i, node in enumerate(all_nodes)}
        
        # Create links
        source_indices = []
        target_indices = []
        values = []
        colors = []
        
        for mapping in field_mappings:
            source_idx = node_indices[mapping["source_field"]]
            target_idx = node_indices[mapping["target_field"]]
            confidence = mapping["confidence"]
            
            source_indices.append(source_idx)
            target_indices.append(target_idx)
            values.append(confidence)
            
            # Color based on confidence
            if confidence >= 0.9:
                colors.append('rgba(0, 255, 0, 0.6)')
            elif confidence >= 0.8:
                colors.append('rgba(255, 255, 0, 0.6)')
            else:
                colors.append('rgba(255, 0, 0, 0.6)')
        
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = all_nodes,
                color = ["lightblue" if i < len(source_fields) else "lightgreen" for i in range(len(all_nodes))]
            ),
            link = dict(
                source = source_indices,
                target = target_indices,
                value = values,
                color = colors
            ))])
        
        fig.update_layout(
            title_text="Field Mapping Flow (Source → Target)",
            font_size=10,
            height=600
        )
        
        return fig
    
    def create_confidence_analysis(self, data):
        """Create confidence analysis charts"""
        mapping_viz = data.get("mapping_visualization", {})
        field_mappings = mapping_viz.get("mapping_details", {}).get("field_mappings", [])
        
        if not field_mappings:
            return go.Figure()
        
        # Extract confidence scores
        confidences = [mapping["confidence"] for mapping in field_mappings]
        source_fields = [mapping["source_field"] for mapping in field_mappings]
        
        # Create histogram
        fig = px.histogram(
            x=confidences,
            nbins=10,
            title="Confidence Score Distribution",
            labels={"x": "Confidence Score", "y": "Number of Mappings"},
            color_discrete_sequence=["steelblue"]
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_mapping_table(self, data):
        """Create detailed mapping table"""
        mapping_viz = data.get("mapping_visualization", {})
        field_mappings = mapping_viz.get("mapping_details", {}).get("field_mappings", [])
        
        if not field_mappings:
            return dash_table.DataTable(data=[])
        
        # Prepare table data
        table_data = []
        for mapping in field_mappings:
            table_data.append({
                "Source Field": mapping["source_field"],
                "Target Field": mapping["target_field"],
                "Data Type": mapping["data_type"],
                "Transformation": mapping["transformation"],
                "Confidence": f"{mapping['confidence']:.0%}"
            })
        
        return dash_table.DataTable(
            data=table_data,
            columns=[{"name": col, "id": col} for col in table_data[0].keys()],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': 'lightgray', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Confidence} >= 90%'},
                    'backgroundColor': '#d4edda',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Confidence} >= 80% && {Confidence} < 90%'},
                    'backgroundColor': '#fff3cd',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Confidence} < 80%'},
                    'backgroundColor': '#f8d7da',
                    'color': 'black',
                }
            ],
            page_size=10,
            sort_action="native",
            filter_action="native"
        )
    
    def create_unmapped_analysis(self, data):
        """Create unmapped fields analysis"""
        mapping_viz = data.get("mapping_visualization", {})
        unmapped_fields = mapping_viz.get("mapping_details", {}).get("unmapped_fields", [])
        
        if not unmapped_fields:
            return html.Div("No unmapped fields"), go.Figure()
        
        # Create table for unmapped fields
        unmapped_table = dash_table.DataTable(
            data=[{"Field": field["source_field"], "Reason": field["reason"]} 
                  for field in unmapped_fields],
            columns=[{"name": "Unmapped Field", "id": "Field"}, 
                    {"name": "Reason", "id": "Reason"}],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': '#f8d7da', 'fontWeight': 'bold'},
            style_data={'backgroundColor': '#f8d7da'}
        )
        
        # Create pie chart for reasons
        reasons = [field["reason"] for field in unmapped_fields]
        reason_counts = pd.Series(reasons).value_counts()
        
        fig = px.pie(
            values=reason_counts.values,
            names=reason_counts.index,
            title="Reasons for Unmapped Fields"
        )
        
        return unmapped_table, fig
    
    def create_transformation_analysis(self, data):
        """Create transformation requirements analysis"""
        mapping_viz = data.get("mapping_visualization", {})
        field_mappings = mapping_viz.get("mapping_details", {}).get("field_mappings", [])
        
        if not field_mappings:
            return go.Figure()
        
        # Count transformation types
        transformations = [mapping["transformation"] for mapping in field_mappings]
        transform_counts = pd.Series(transformations).value_counts()
        
        fig = px.pie(
            values=transform_counts.values,
            names=transform_counts.index,
            title="Transformation Types Required",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        return fig
    
    def create_source_target_overview(self, data):
        """Create source vs target overview section"""
        mapping_viz = data.get("mapping_visualization", {})
        source = mapping_viz.get("source_structure", {})
        target = mapping_viz.get("target_structure", {})
        
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Source File"),
                    dbc.CardBody([
                        html.H5(source.get("file_name", "N/A"), className="card-title"),
                        html.P(f"Type: {source.get('file_type', 'N/A')}", className="card-text"),
                        html.P(f"Rows: {source.get('total_rows', 0):,}", className="card-text"),
                        html.P(f"Columns: {source.get('total_columns', 0)}", className="card-text"),
                        html.P(f"Domain: {source.get('domain_detected', 'N/A')}", className="card-text")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Target Table"),
                    dbc.CardBody([
                        html.H5(target.get("table_name", "N/A"), className="card-title"),
                        html.P(f"Code: {target.get('table_code', 'N/A')}", className="card-text"),
                        html.P(f"Confidence: {target.get('confidence_score', 0):.0%}", className="card-text"),
                        html.P(target.get("selection_reason", "N/A")[:100] + "...", className="card-text small")
                    ])
                ])
            ], width=6)
        ])
    
    def create_dashboard_layout(self, initial_file=None):
        """Create the dynamic dashboard layout"""
        # Load initial data
        if initial_file:
            mapping_data = self.load_mapping_data(initial_file)
        else:
            latest_file = self.get_latest_analysis_file()
            mapping_data = self.load_mapping_data(latest_file) if latest_file else self.sample_data
        
        return dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🎯 Data Mapping Visualization Dashboard", className="text-center mb-2"),
                    html.P("Interactive visualization of agent analysis results", 
                          className="text-center text-muted mb-4"),
                    html.Hr()
                ])
            ]),
            
            # File selector
            html.Div(id="file-selector-container"),
            
            # Dynamic content container
            html.Div(id="dashboard-content"),
            
            # Auto-refresh interval (every 30 seconds to check for new files)
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # 30 seconds
                n_intervals=0
            )
            
        ], fluid=True)
    
    def create_dashboard_content(self, mapping_data):
        """Create the main dashboard content"""
        source_name = mapping_data['mapping_visualization']['source_structure']['file_name']
        
        return [
            # Status banner
            dbc.Alert([
                html.H5(f"📊 Currently Analyzing: {source_name}", className="mb-1"),
                html.P(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", className="mb-0 small")
            ], color="info", className="mb-4"),
            
            # KPI Cards
            html.H3("Executive Summary", className="mt-4 mb-3"),
            self.create_kpi_cards(mapping_data),
            
            html.Hr(),
            
            # Source vs Target Overview
            html.H4("Source vs Target Overview", className="mb-3"),
            self.create_source_target_overview(mapping_data),
            
            html.Hr(),
            
            # Main visualizations
            dbc.Row([
                dbc.Col([
                    html.H4("Field Mapping Flow"),
                    dcc.Graph(
                        figure=self.create_sankey_diagram(mapping_data),
                        id="sankey-diagram"
                    )
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Confidence Analysis"),
                    dcc.Graph(
                        figure=self.create_confidence_analysis(mapping_data),
                        id="confidence-chart"
                    )
                ], width=6),
                dbc.Col([
                    html.H4("Transformation Requirements"),
                    dcc.Graph(
                        figure=self.create_transformation_analysis(mapping_data),
                        id="transformation-chart"
                    )
                ], width=6)
            ], className="mb-4"),
            
            # Detailed tables
            html.H4("Detailed Field Mappings"),
            self.create_mapping_table(mapping_data),
            
            html.Hr(),
            
            # Unmapped fields analysis
            html.H4("Unmapped Fields Analysis", className="mt-4"),
            html.Div(id="unmapped-content")
        ]

# Initialize dashboard
dashboard = MappingDashboard()

# Create the app layout
app.layout = dashboard.create_dashboard_layout()

# Callback for file selector initialization
@app.callback(
    Output("file-selector-container", "children"),
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=False
)
def update_file_selector(n_intervals):
    return dashboard.create_file_selector_dropdown()

# Main callback for updating dashboard content
@app.callback(
    Output("dashboard-content", "children"),
    [Input("file-selector-container", "children")],
    prevent_initial_call=False
)
def update_dashboard_content(file_selector_children):
    # Get the latest file as default
    latest_file = dashboard.get_latest_analysis_file()
    if not latest_file:
        return html.Div("No analysis files found. Run the agent on some data files first!")
    
    try:
        mapping_data = dashboard.load_mapping_data(latest_file)
        return dashboard.create_dashboard_content(mapping_data)
    except Exception as e:
        return dbc.Alert(f"Error loading analysis file: {str(e)}", color="danger")

# Secondary callback for when user actually selects a file
@app.callback(
    Output("dashboard-content", "children", allow_duplicate=True),
    [Input("file-selector", "value")],
    prevent_initial_call=True
)
def update_dashboard_on_selection(selected_file):
    if not selected_file:
        return html.Div("Please select an analysis file to visualize.")
    
    try:
        mapping_data = dashboard.load_mapping_data(selected_file)
        return dashboard.create_dashboard_content(mapping_data)
    except Exception as e:
        return dbc.Alert(f"Error loading analysis file: {str(e)}", color="danger")

# Callback for unmapped fields (needs to be dynamic now)
@app.callback(
    Output("unmapped-content", "children"),
    [Input("file-selector", "value")]
)
def update_unmapped_analysis(selected_file):
    if not selected_file:
        return html.Div("No file selected")
    
    try:
        mapping_data = dashboard.load_mapping_data(selected_file)
        table, chart = dashboard.create_unmapped_analysis(mapping_data)
        return dbc.Row([
            dbc.Col([table], width=6),
            dbc.Col([dcc.Graph(figure=chart)], width=6)
        ])
    except:
        return html.Div("Error loading unmapped analysis")

def start_dashboard(port=8050, debug=True):
    """Start the dashboard server"""
    files = dashboard.get_available_analysis_files()
    
    print("Starting Dynamic Data Mapping Dashboard...")
    print(f"Dashboard available at: http://localhost:{port}")
    print(f"Monitoring directory: {dashboard.results_dir}")
    print(f"Found {len(files)} analysis files:")
    
    for file_info in files[:5]:  # Show first 5
        print(f"   • {file_info['source_file']} ({file_info['modified_time'].strftime('%Y-%m-%d %H:%M')})")
    
    if len(files) > 5:
        print(f"   ... and {len(files) - 5} more files")
    
    if not files:
        print("No analysis files found. Run the agent on some data files first!")
    
    print("\nDashboard features:")
    print("Select any analysis file from dropdown")
    print("Auto-refreshes every 30 seconds for new analyses")
    print("Click 'Refresh Files' button for immediate update")
    
    app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    start_dashboard()