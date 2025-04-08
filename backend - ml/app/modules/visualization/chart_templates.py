import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json

class ChartTemplates:
    """
    Predefined chart templates for business visualizations
    """
    
    @staticmethod
    def financial_summary(revenue_data: pd.DataFrame, expense_data: pd.DataFrame, 
                          date_column: str, revenue_column: str, expense_column: str) -> Dict[str, Any]:
        """Create a financial summary dashboard with revenue, expenses, and profit"""
        # Ensure data is sorted by date
        revenue_data = revenue_data.sort_values(date_column)
        expense_data = expense_data.sort_values(date_column)
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add revenue line
        fig.add_trace(
            go.Scatter(
                x=revenue_data[date_column], 
                y=revenue_data[revenue_column],
                name="Revenue",
                line=dict(color="#1f77b4", width=3)
            ),
            secondary_y=False,
        )
        
        # Add expense line
        fig.add_trace(
            go.Scatter(
                x=expense_data[date_column], 
                y=expense_data[expense_column],
                name="Expenses",
                line=dict(color="#ff7f0e", width=3)
            ),
            secondary_y=False,
        )
        
        # Calculate profit
        profit_data = pd.merge(revenue_data, expense_data, on=date_column)
        profit_data['profit'] = profit_data[revenue_column] - profit_data[expense_column]
        
        # Add profit bars
        fig.add_trace(
            go.Bar(
                x=profit_data[date_column],
                y=profit_data['profit'],
                name="Profit",
                marker_color=profit_data['profit'].apply(
                    lambda x: 'green' if x >= 0 else 'red'
                )
            ),
            secondary_y=True,
        )
        
        # Set figure layout
        fig.update_layout(
            title_text="Financial Performance Summary",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=600,
        )
        
        # Set x-axis title
        fig.update_xaxes(title_text="Date")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Amount", secondary_y=False)
        fig.update_yaxes(title_text="Profit", secondary_y=True)
        
        # Calculate summary statistics
        total_revenue = revenue_data[revenue_column].sum()
        total_expenses = expense_data[expense_column].sum()
        total_profit = total_revenue - total_expenses
        profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        summary_stats = {
            'total_revenue': float(total_revenue),
            'total_expenses': float(total_expenses),
            'total_profit': float(total_profit),
            'profit_margin': float(profit_margin),
            'revenue_trend': float(revenue_data[revenue_column].iloc[-1] - revenue_data[revenue_column].iloc[0]),
            'expense_trend': float(expense_data[expense_column].iloc[-1] - expense_data[expense_column].iloc[0])
        }
        
        return {
            'type': 'financial_summary',
            'title': 'Financial Performance Summary',
            'plotly_figure': fig.to_json(),
            'summary_stats': summary_stats
        }
    
    @staticmethod
    def sales_by_region(sales_data: pd.DataFrame, region_column: str, 
                       sales_column: str, date_column: Optional[str] = None) -> Dict[str, Any]:
        """Create a regional sales comparison visualization"""
        # Group by region
        if date_column:
            # Get the most recent date for each region
            latest_date = sales_data[date_column].max()
            recent_data = sales_data[sales_data[date_column] == latest_date]
            regional_sales = recent_data.groupby(region_column)[sales_column].sum().reset_index()
        else:
            regional_sales = sales_data.groupby(region_column)[sales_column].sum().reset_index()
        
        # Sort by sales value
        regional_sales = regional_sales.sort_values(sales_column, ascending=False)
        
        # Create bar chart
        fig = px.bar(
            regional_sales,
            x=region_column,
            y=sales_column,
            title="Sales by Region",
            color=sales_column,
            color_continuous_scale="Blues",
            text=sales_column
        )
        
        fig.update_traces(
            texttemplate='%{text:.2s}', 
            textposition='outside'
        )
        
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Region",
            yaxis_title="Sales",
            coloraxis_showscale=False,
            height=500
        )
        
        # Create map visualization if coordinates available
        map_fig = None
        if 'latitude' in sales_data.columns and 'longitude' in sales_data.columns:
            map_data = sales_data.groupby([region_column, 'latitude', 'longitude'])[sales_column].sum().reset_index()
            
            map_fig = px.scatter_geo(
                map_data,
                lat='latitude',
                lon='longitude',
                size=sales_column,
                color=sales_column,
                hover_name=region_column,
                title="Geographic Sales Distribution",
                color_continuous_scale="Blues",
                size_max=50
            )
            
            map_fig.update_layout(
                template="plotly_white",
                height=500
            )
        
        # Calculate summary statistics
        top_region = regional_sales.iloc[0][region_column]
        top_region_sales = float(regional_sales.iloc[0][sales_column])
        bottom_region = regional_sales.iloc[-1][region_column]
        bottom_region_sales = float(regional_sales.iloc[-1][sales_column])
        
        # Calculate disparity ratio between top and bottom regions
        disparity_ratio = top_region_sales / bottom_region_sales if bottom_region_sales > 0 else float('inf')
        
        summary_stats = {
            'top_region': top_region,
            'top_region_sales': top_region_sales,
            'bottom_region': bottom_region,
            'bottom_region_sales': bottom_region_sales,
            'disparity_ratio': float(disparity_ratio),
            'total_regions': len(regional_sales),
            'total_sales': float(regional_sales[sales_column].sum())
        }
        
        return {
            'type': 'sales_by_region',
            'title': 'Sales by Region',
            'bar_chart': fig.to_json(),
            'map_chart': map_fig.to_json() if map_fig else None,
            'summary_stats': summary_stats
        }
    
    @staticmethod
    def customer_segmentation(customer_data: pd.DataFrame, value_column: str, 
                             frequency_column: str, recency_column: str) -> Dict[str, Any]:
        """Create RFM (Recency, Frequency, Monetary Value) customer segmentation"""
        # Create RFM segments
        # Recency: lower is better (more recent)
        # Frequency: higher is better
        # Monetary: higher is better
        
        rfm_data = customer_data.copy()
        
        # Create quintiles for each dimension
        rfm_data['R_quintile'] = pd.qcut(rfm_data[recency_column], 5, labels=False, duplicates='drop')
        rfm_data['F_quintile'] = pd.qcut(rfm_data[frequency_column], 5, labels=False, duplicates='drop')
        rfm_data['M_quintile'] = pd.qcut(rfm_data[value_column], 5, labels=False, duplicates='drop')
        
        # Reverse recency (lower is better)
        rfm_data['R_quintile'] = 4 - rfm_data['R_quintile']
        
        # Calculate RFM score
        rfm_data['RFM_score'] = rfm_data['R_quintile'] + rfm_data['F_quintile'] + rfm_data['M_quintile']
        
        # Create segments
        def segment_customer(score):
            if score >= 10:
                return 'Champions'
            elif score >= 8:
                return 'Loyal Customers'
            elif score >= 6:
                return 'Potential Loyalists'
            elif score >= 4:
                return 'At Risk Customers'
            else:
                return 'Needs Attention'
        
        rfm_data['Segment'] = rfm_data['RFM_score'].apply(segment_customer)
        
        # Create segment summary
        segment_summary = rfm_data.groupby('Segment').agg({
            value_column: ['mean', 'sum', 'count'],
            frequency_column: 'mean',
            recency_column: 'mean'
        }).reset_index()
        
        segment_summary.columns = ['Segment', 'Avg_Value', 'Total_Value', 'Count', 'Avg_Frequency', 'Avg_Recency']
        segment_summary = segment_summary.sort_values('Total_Value', ascending=False)
        
        # Create pie chart of customer segments
        pie_fig = px.pie(
            rfm_data,
            names='Segment',
            title="Customer Segmentation",
            color='Segment',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        pie_fig.update_layout(
            template="plotly_white",
            legend_title="Segment"
        )
        
        # Create bubble chart of segments
        bubble_fig = px.scatter(
            segment_summary,
            x='Avg_Recency',
            y='Avg_Frequency',
            size='Total_Value',
            color='Segment',
            text='Segment',
            title="Customer Segment Analysis",
            color_discrete_sequence=px.colors.qualitative.Set3,
            size_max=60
        )
        
        bubble_fig.update_layout(
            template="plotly_white",
            xaxis_title="Average Recency (days)",
            yaxis_title="Average Frequency (purchases)",
            legend_title="Segment"
        )
        
        # Create 3D scatter plot of individual customers
        scatter3d_fig = px.scatter_3d(
            rfm_data,
            x=recency_column,
            y=frequency_column,
            z=value_column,
            color='Segment',
            title="3D RFM Customer Visualization",
            color_discrete_sequence=px.colors.qualitative.Set3,
            opacity=0.7
        )
        
        scatter3d_fig.update_layout(
            template="plotly_white",
            scene=dict(
                xaxis_title="Recency",
                yaxis_title="Frequency",
                zaxis_title="Monetary Value"
            )
        )
        
        # Calculate summary statistics
        total_customers = len(rfm_data)
        total_value = float(rfm_data[value_column].sum())
        avg_value_per_customer = float(rfm_data[value_column].mean())
        
        top_segment = segment_summary.iloc[0]['Segment']
        top_segment_pct = float(segment_summary.iloc[0]['Count'] / total_customers * 100)
        top_segment_value = float(segment_summary.iloc[0]['Total_Value'])
        top_segment_value_pct = float(top_segment_value / total_value * 100)
        
        summary_stats = {
            'total_customers': total_customers,
            'total_value': total_value,
            'avg_value_per_customer': avg_value_per_customer,
            'top_segment': top_segment,
            'top_segment_pct': top_segment_pct,
            'top_segment_value': top_segment_value,
            'top_segment_value_pct': top_segment_value_pct,
            'segment_counts': rfm_data['Segment'].value_counts().to_dict()
        }
        
        return {
            'type': 'customer_segmentation',
            'title': 'Customer Segmentation Analysis',
            'pie_chart': pie_fig.to_json(),
            'bubble_chart': bubble_fig.to_json(),
            'scatter3d_chart': scatter3d_fig.to_json(),
            'segment_summary': segment_summary.to_dict(orient='records'),
            'summary_stats': summary_stats
        }
    
    @staticmethod
    def product_performance_matrix(product_data: pd.DataFrame, product_column: str,
                                  revenue_column: str, growth_column: str) -> Dict[str, Any]:
        """Create BCG matrix style product performance visualization"""
        # Calculate median values for quadrant division
        revenue_median = product_data[revenue_column].median()
        growth_median = product_data[growth_column].median()
        
        # Assign quadrants
        def assign_quadrant(row):
            if row[revenue_column] >= revenue_median and row[growth_column] >= growth_median:
                return "Stars"
            elif row[revenue_column] >= revenue_median and row[growth_column] < growth_median:
                return "Cash Cows"
            elif row[revenue_column] < revenue_median and row[growth_column] >= growth_median:
                return "Question Marks"
            else:
                return "Dogs"
        
        product_data['Quadrant'] = product_data.apply(assign_quadrant, axis=1)
        
        # Create scatter plot
        fig = px.scatter(
            product_data,
            x=revenue_column,
            y=growth_column,
            color='Quadrant',
            size=revenue_column,
            hover_name=product_column,
            title="Product Performance Matrix",
            color_discrete_map={
                'Stars': '#FFC107',       # Yellow
                'Cash Cows': '#4CAF50',   # Green
                'Question Marks': '#2196F3', # Blue
                'Dogs': '#F44336'         # Red
            },
            size_max=50
        )
        
        # Add quadrant dividers
        fig.add_hline(y=growth_median, line_dash="dash", line_color="gray")
        fig.add_vline(x=revenue_median, line_dash="dash", line_color="gray")
        
        # Add quadrant labels
        fig.add_annotation(x=revenue_median*1.5, y=growth_median*1.5, text="Stars",
                          showarrow=False, font=dict(size=16, color="#FFC107"))
        fig.add_annotation(x=revenue_median*1.5, y=growth_median*0.5, text="Cash Cows",
                          showarrow=False, font=dict(size=16, color="#4CAF50"))
        fig.add_annotation(x=revenue_median*0.5, y=growth_median*1.5, text="Question Marks",
                          showarrow=False, font=dict(size=16, color="#2196F3"))
        fig.add_annotation(x=revenue_median*0.5, y=growth_median*0.5, text="Dogs",
                          showarrow=False, font=dict(size=16, color="#F44336"))
        
        fig.update_layout(
            template="plotly_white",
            xaxis_title=revenue_column,
            yaxis_title=growth_column,
            legend_title="Quadrant",
            height=600
        )
        
        # Create summary table by quadrant
        quadrant_summary = product_data.groupby('Quadrant').agg({
            product_column: 'count',
            revenue_column: 'sum',
            growth_column: 'mean'
        }).reset_index()
        
        quadrant_summary.columns = ['Quadrant', 'Product_Count', 'Total_Revenue', 'Avg_Growth']
        
        # Calculate percentage of total revenue by quadrant
        total_revenue = quadrant_summary['Total_Revenue'].sum()
        quadrant_summary['Revenue_Percentage'] = quadrant_summary['Total_Revenue'] / total_revenue * 100
        
        # Calculate summary statistics
        summary_stats = {
            'total_products': len(product_data),
            'total_revenue': float(total_revenue),
            'quadrant_counts': product_data['Quadrant'].value_counts().to_dict(),
            'quadrant_revenue': {
                row['Quadrant']: {
                    'total': float(row['Total_Revenue']),
                    'percentage': float(row['Revenue_Percentage'])
                } for _, row in quadrant_summary.iterrows()
            }
        }
        
        return {
            'type': 'product_performance_matrix',
            'title': 'Product Performance Matrix',
            'matrix_chart': fig.to_json(),
            'quadrant_summary': quadrant_summary.to_dict(orient='records'),
            'summary_stats': summary_stats
        }
    
    @staticmethod
    def market_share_analysis(market_data: pd.DataFrame, company_column: str, 
                             share_column: str, period_column: Optional[str] = None) -> Dict[str, Any]:
        """Create market share analysis visualization"""
        # If period column provided, get latest period
        if period_column:
            latest_period = market_data[period_column].max()
            current_data = market_data[market_data[period_column] == latest_period]
        else:
            current_data = market_data
        
        # Sort by market share
        current_data = current_data.sort_values(share_column, ascending=False)
        
        # Create pie chart
        pie_fig = px.pie(
            current_data,
            names=company_column,
            values=share_column,
            title="Market Share Distribution",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        pie_fig.update_layout(
            template="plotly_white",
            legend_title=company_column
        )
        
        # If period data available, create time series
        line_fig = None
        if period_column:
            # Pivot data for time series
            pivot_data = market_data.pivot(index=period_column, columns=company_column, values=share_column)
            
            # Create line chart
            line_fig = px.line(
                pivot_data,
                x=pivot_data.index,
                y=pivot_data.columns,
                title="Market Share Trends Over Time",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            line_fig.update_layout(
                template="plotly_white",
                xaxis_title=period_column,
                yaxis_title="Market Share (%)",
                legend_title=company_column
            )
        
        # Calculate concentration metrics
        # Herfindahl-Hirschman Index (HHI) - measure of market concentration
        hhi = ((current_data[share_column] / 100) ** 2).sum() * 10000
        
        # CR4 - Combined share of top 4 companies
        cr4 = current_data.head(4)[share_column].sum()
        
        # Calculate summary statistics
        leader = current_data.iloc[0][company_column]
        leader_share = float(current_data.iloc[0][share_column])
        
        runner_up = current_data.iloc[1][company_column] if len(current_data) > 1 else None
        runner_up_share = float(current_data.iloc[1][share_column]) if len(current_data) > 1 else 0
        
        # Leadership gap
        leadership_gap = leader_share - runner_up_share
        
        summary_stats = {
            'total_companies': len(current_data),
            'market_leader': leader,
            'leader_share': leader_share,
            'runner_up': runner_up,
            'runner_up_share': runner_up_share,
            'leadership_gap': leadership_gap,
            'hhi': float(hhi),
            'cr4': float(cr4),
            'market_concentration': 'High' if hhi > 2500 else ('Moderate' if hhi > 1500 else 'Low')
        }
        
        return {
            'type': 'market_share_analysis',
            'title': 'Market Share Analysis',
            'pie_chart': pie_fig.to_json(),
            'line_chart': line_fig.to_json() if line_fig else None,
            'summary_stats': summary_stats
        }