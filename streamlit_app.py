import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

# --- Helper Functions (Our Logic from Stages 1-4) ---

def run_stage_1_upload(uploaded_files):
    """
    Takes a list of Streamlit UploadedFile objects and runs the
    Stage 1 identification logic.
    """
    df_inventory_cogs = None
    df_inventory_history = None
    df_bin_info = None

    if len(uploaded_files) != 3:
        st.error(f"Please upload all 3 files. You uploaded {len(uploaded_files)}.")
        return None, None, None

    for file in uploaded_files:
        st.write(f"\nProcessing file: {file.name}...")

        # Read file content as bytes
        content = file.getvalue()

        # Try to identify based on content
        try:
            # Try 'utf-8', fallback to 'latin1'
            try:
                file_content_str = content.decode('utf-8')
            except UnicodeDecodeError:
                file_content_str = content.decode('latin1')

            # --- Identify Files Based on Unique Column Names/Content ---

            # File 1: Inventory & COGS
            if 'Item/COGS' in file_content_str and 'Sep 2025' in file_content_str:
                st.write("File identified as: Inventory & COGS")
                df = pd.read_csv(
                    io.StringIO(file_content_str),
                    header=1,
                    on_bad_lines='warn',
                    delimiter=',',
                    skipinitialspace=True
                )
                df.columns = df.columns.str.strip()
                df_inventory_cogs = df

            # File 2: Inventory History
            elif 'Oct 2024' in file_content_str and 'Total Quantity On Hand' in file_content_str:
                st.write("File identified as: Inventory History")
                df = pd.read_csv(
                    io.StringIO(file_content_str),
                    header=0,
                    on_bad_lines='warn',
                    delimiter=',',
                    skipinitialspace=True
                )
                df.columns = df.columns.str.strip()
                df_inventory_history = df

            # File 3: DC Bin Info
            elif 'Location Class' in file_content_str and 'Location' in file_content_str:
                st.write("File identified as: DC Bin Info")
                df = pd.read_csv(
                    io.StringIO(file_content_str),
                    header=0,
                    skiprows=[1, 2], # Skip the junk rows
                    on_bad_lines='warn',
                    delimiter=',',
                    skipinitialspace=True
                )
                df.columns = df.columns.str.strip()
                df_bin_info = df

            else:
                st.warning(f"Could not identify file: {file.name}. Please check the file content.")

        except Exception as e:
            st.error(f"An error occurred while processing {file.name}: {e}")

    # --- Verification Check ---
    if df_inventory_cogs is None or df_inventory_history is None or df_bin_info is None:
        st.error("One or more files was not identified. Please ensure you uploaded the 3 correct files.")
        return None, None, None

    st.success("Stage 1 Complete: All 3 files successfully loaded and identified.")
    return df_inventory_cogs, df_inventory_history, df_bin_info

def run_stage_2_cleaning(df_inventory_cogs, df_inventory_history, df_bin_info):
    """
    Runs the full Stage 2 cleaning logic on the DataFrames.
    """
    st.write("--- Starting Stage 2: Data Cleaning ---")

    # --- 1. Clean Inventory & COGS ---
    try:
        st.write("\nCleaning Inventory & COGS...")
        original_rows = len(df_inventory_cogs)
        df_cogs_clean = df_inventory_cogs.copy()
        df_cogs_clean = df_cogs_clean[df_cogs_clean['Code'] != 'TOTAL']
        key_cols = ['Code', 'Product Description']
        numeric_cols = ['Length', 'Width', 'Height', 'Quantity On Hand', 'Item/COGS']
        df_cogs_clean.dropna(subset=key_cols, inplace=True)
        for col in numeric_cols:
            df_cogs_clean[col] = df_cogs_clean[col].astype(str).str.replace(',', '', regex=False)
            df_cogs_clean[col] = pd.to_numeric(df_cogs_clean[col], errors='coerce')
        df_cogs_clean.dropna(subset=numeric_cols, inplace=True)
        st.write(f"   > COGS cleaning complete. Final rows: {len(df_cogs_clean)}")
    except Exception as e:
        st.error(f"An error occurred cleaning Inventory & COGS: {e}")
        return None, None, None

    # --- 2. Clean Inventory History ---
    try:
        st.write("\nCleaning Inventory History...")
        df_history_clean = df_inventory_history.copy()
        df_history_clean = df_history_clean[df_history_clean['Code'] != 'TOTAL']
        key_cols = ['Code', 'Product Description']
        numeric_cols = [col for col in df_history_clean.columns if col not in key_cols]
        df_history_clean.dropna(subset=key_cols, inplace=True)
        for col in numeric_cols:
            df_history_clean[col] = df_history_clean[col].astype(str).str.replace(',', '', regex=False)
            df_history_clean[col] = pd.to_numeric(df_history_clean[col], errors='coerce')
        df_history_clean.dropna(subset=numeric_cols, inplace=True)
        st.write(f"   > History cleaning complete. Final rows: {len(df_history_clean)}")
    except Exception as e:
        st.error(f"An error occurred cleaning Inventory History: {e}")
        return None, None, None

    # --- 3. Clean DC Bin Info ---
    try:
        st.write("\nCleaning DC Bin Info...")
        df_bin_clean = df_bin_info.copy()

        # Step 1-2: Drop blanks, create group
        df_bin_clean.dropna(subset=['Location'], inplace=True)
        df_bin_clean['Location_Group'] = df_bin_clean['Location'].str.rsplit('-', n=1).str[0]

        # Step 3-4: Convert & Inherit Dimensions
        dimension_cols = ['Length', 'Width', 'Height']
        for col in dimension_cols:
            df_bin_clean[col] = pd.to_numeric(df_bin_clean[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
            group_value = df_bin_clean.groupby('Location_Group')[col].transform('first')
            df_bin_clean[col] = df_bin_clean[col].fillna(group_value)
        df_bin_clean[dimension_cols] = df_bin_clean[dimension_cols].fillna(0)

        # Step 5: Calculate Group Cubic Volume
        df_bin_clean['Group_Cubic_Volume'] = df_bin_clean['Length'] * df_bin_clean['Width'] * df_bin_clean['Height']

        # Step 6: Calculate Volume per Partition
        partition_counts = df_bin_clean.groupby('Location_Group')['Location'].transform('count')
        df_bin_clean['Partition_Count'] = partition_counts
        df_bin_clean['Partition_Cubic_Volume'] = df_bin_clean['Group_Cubic_Volume'] / df_bin_clean['Partition_Count']
        st.write(f"   > Calculated partition volumes.")

        # Step 7: Remove Outlier Bins
        original_bin_rows = len(df_bin_clean)
        max_dimension = 200
        df_bin_clean = df_bin_clean[
            (df_bin_clean['Length'] <= max_dimension) &
            (df_bin_clean['Width'] <= max_dimension) &
            (df_bin_clean['Height'] <= max_dimension)
        ]
        filtered_bin_rows = len(df_bin_clean)
        st.write(f"   > Removed {original_bin_rows - filtered_bin_rows} outlier bins.")

        # Step 8: Convert dummy columns
        dummy_cols = ['No Putaway', 'Product Limit']
        for col in dummy_cols:
            if col in df_bin_clean.columns:
                df_bin_clean[col] = pd.to_numeric(df_bin_clean[col], errors='coerce')
                df_bin_clean[col] = df_bin_clean[col].fillna(0)

        st.write(f"   > Bin cleaning complete. Final rows: {len(df_bin_clean)}")

    except Exception as e:
        st.error(f"An error occurred cleaning DC Bin Info: {e}")
        return None, None, None

    st.success("Stage 2 Complete: All 3 files have been cleaned.")
    return df_cogs_clean, df_history_clean, df_bin_clean

def run_stage_3_analysis(df_cogs_clean, df_history_clean, df_bin_clean):
    """
    Runs the Stage 3 analysis and returns a summary DataFrame
    and the processed DataFrames for export.
    """
    st.write("--- Starting Stage 3: Analysis & Metrics ---")

    try:
        cogs_df = df_cogs_clean.copy()
        history_df = df_history_clean.copy()
        bin_df = df_bin_clean.copy()

        # Metric 1: Total Inventory Value
        cogs_df['Current_Inventory_Value'] = cogs_df['Quantity On Hand'] * cogs_df['Item/COGS']
        total_inventory_value = cogs_df['Current_Inventory_Value'].sum()

        # Metric 2: Inventory Turnover Rate
        turnover_df = pd.merge(
            cogs_df[['Code', 'Item/COGS', 'Current_Inventory_Value']],
            history_df[['Code', 'Total Quantity On Hand']],
            on='Code',
            how='inner'
        )
        if turnover_df.empty:
            st.warning("No matching 'Code' for turnover calculation.")
            total_annual_cogs = 0.0
            inventory_turnover_rate = 0.0
        else:
            turnover_df.rename(columns={'Total Quantity On Hand': 'Annual_Sales_Units'}, inplace=True)
            turnover_df['Annual_COGS'] = turnover_df['Annual_Sales_Units'] * turnover_df['Item/COGS']
            total_annual_cogs = turnover_df['Annual_COGS'].sum()
            total_current_inventory_value_for_turnover = turnover_df['Current_Inventory_Value'].sum()
            inventory_turnover_rate = total_annual_cogs / total_current_inventory_value_for_turnover if total_current_inventory_value_for_turnover > 0 else 0

        # Metric 3: Warehouse Fill Percentage
        cogs_df['Product_Cubic_Volume'] = cogs_df['Length'] * cogs_df['Width'] * cogs_df['Height']
        cogs_df['Total_Product_Volume'] = cogs_df['Product_Cubic_Volume'] * cogs_df['Quantity On Hand']
        total_inventory_volume = cogs_df['Total_Product_Volume'].sum()

        total_bin_volume = bin_df['Partition_Cubic_Volume'].sum()

        volume_loss_percent = 0.115 # 11.5%
        effective_bin_volume = total_bin_volume * (1 - volume_loss_percent)

        warehouse_fill_percentage = (total_inventory_volume / effective_bin_volume) * 100 if effective_bin_volume > 0 else 0

        # Consolidate Results
        st.write("   > Analysis complete.")
        metrics = {
            'Total Current Inventory Value': f"${total_inventory_value:,.2f}",
            'Total Annual COGS (Based on 12-mo sales)': f"${total_annual_cogs:,.2f}",
            'Inventory Turnover Rate': f"{inventory_turnover_rate:.2f} (Annual COGS / Current Value)",
            '---': '---',
            'Total Inventory Volume (Cubic)': f"{total_inventory_volume:,.2f}",
            'Total Bin Volume (Cubic, Raw)': f"{total_bin_volume:,.2f} ({len(bin_df)} total partitions)",
            'Stacking Inefficiency Loss (11.5%)': f"-{total_bin_volume * volume_loss_percent:,.2f}",
            'Effective Bin Volume (Cubic, Usable)': f"{effective_bin_volume:,.2f}",
            'Warehouse Fill Percentage (Effective)': f"{warehouse_fill_percentage:.2f}%"
        }
        df_analysis_summary = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

        st.success("Stage 3 Complete: Analysis finished.")

        # Return all DataFrames needed for Stage 3.2 and 4
        return df_analysis_summary, cogs_df, bin_df, turnover_df

    except Exception as e:
        st.error(f"An error occurred during Stage 3 analysis: {e}")
        return None, None, None, None

def run_stage_3_2_graphs(cogs_df, bin_df):
    """
    Runs the Stage 3.2 visualization logic.
    """
    st.write("--- Starting Stage 3.2: Generating Visualizations ---")

    try:
        # --- 1. Bin Classification ---
        bin_cutoffs = [0, 1728, 1728 * 8, 1728 * 27, float('inf')]
        bin_labels = ['Small', 'Medium', 'Large', 'Bulk']
        bin_df['Bin_Type'] = pd.cut(bin_df['Partition_Cubic_Volume'], bins=bin_cutoffs, labels=bin_labels, right=False)
        bin_type_counts = bin_df['Bin_Type'].value_counts().sort_index()

        # --- 2. Item Classification ---
        cogs_df['Item_Type'] = pd.cut(cogs_df['Product_Cubic_Volume'], bins=bin_cutoffs, labels=bin_labels, right=False)
        item_type_counts = cogs_df['Item_Type'].value_counts().sort_index()

        # --- 3. Top 10 Analysis ---
        top_10_value = cogs_df.nlargest(10, 'Current_Inventory_Value').set_index('Product Description')['Current_Inventory_Value']
        top_10_volume = cogs_df.nlargest(10, 'Total_Product_Volume').set_index('Product Description')['Total_Product_Volume']

        # --- 4. Generate Graphs ---
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
        fig.suptitle('Warehouse Analysis Dashboard', fontsize=24, fontweight='bold')
        plt.subplots_adjust(hspace=0.4, wspace=0.2)

        # Graph 1: Bin Size Distribution
        ax1 = axes[0, 0]
        bin_type_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title('Graph 1: Bin Size Distribution', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Bin Type (by Partition Volume)')
        ax1.set_ylabel('Count of Partitions')
        ax1.tick_params(axis='x', rotation=45)
        for i, count in enumerate(bin_type_counts):
            ax1.text(i, count + 50, f'{count:,}', ha='center', fontsize=10)

        # Graph 2: Item Size Distribution
        ax2 = axes[0, 1]
        item_type_counts.plot(kind='bar', ax=ax2, color='lightgreen', edgecolor='black')
        ax2.set_title('Graph 2: Item Size Distribution', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Item Type (by Product Volume)')
        ax2.set_ylabel('Count of Unique Products (SKUs)')
        ax2.tick_params(axis='x', rotation=45)
        for i, count in enumerate(item_type_counts):
            ax2.text(i, count + 50, f'{count:,}', ha='center', fontsize=10)

        # Graph 3: Top 10 Most Valuable Items
        ax3 = axes[1, 0]
        top_10_value.plot(kind='barh', ax=ax3, color='gold', edgecolor='black')
        ax3.set_title('Graph 3: Top 10 Most Valuable Items', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Total Inventory Value ($)')
        ax3.set_ylabel('Product Description')
        ax3.invert_yaxis()
        for i, (value, name) in enumerate(zip(top_10_value, top_10_value.index)):
            ax3.text(value, i, f' ${value:,.0f}', va='center', fontsize=10)

        # Graph 4: Top 10 Space-Hog Items
        ax4 = axes[1, 1]
        top_10_volume.plot(kind='barh', ax=ax4, color='salmon', edgecolor='black')
        ax4.set_title('Graph 4: Top 10 "Space Hog" Items', fontsize=16, fontweight='bold')
        ax4.set_xlabel('Total Inventory Volume (Cubic Inches)')
        ax4.set_ylabel('Product Description')
        ax4.invert_yaxis()
        for i, (value, name) in enumerate(zip(top_10_volume, top_10_volume.index)):
            ax4.text(value, i, f' {value:,.0f} cu in', va='center', fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        st.success("Stage 3.2 Complete: Graphs generated.")

        # Return the figure object to display
        return fig, cogs_df, bin_df

    except Exception as e:
        st.error(f"An error occurred during Stage 3.2 analysis: {e}")
        return None, cogs_df, bin_df

def run_stage_4_export(df_analysis_summary, df_volume_details, df_bin_info_processed, df_turnover_details):
    """
    Runs the Stage 4 export logic and returns the Excel file as bytes.
    """
    st.write("--- Starting Stage 4: Generating Output File ---")

    try:
        # Create the Excel file in memory
        output_io = io.BytesIO()
        with pd.ExcelWriter(output_io, engine='openpyxl') as writer:
            df_analysis_summary.to_excel(writer, sheet_name='Analysis_Summary', index=True)
            df_volume_details.to_excel(writer, sheet_name='Volume_Details', index=False)
            df_bin_info_processed.to_excel(writer, sheet_name='Bin_Info_Processed', index=False)
            df_turnover_details.to_excel(writer, sheet_name='Turnover_Details', index=False)

        st.success("Stage 4 Complete: Master file ready for download.")

        # Get the bytes from the in-memory file
        return output_io.getvalue()

    except Exception as e:
        st.error(f"An error occurred during Stage 4 file export: {e}")
        return None

# --- Main Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("Marcone Warehouse Analysis Tool")

st.info("Please upload the 3 required data files: 'Inventory & COGS', 'Inventory History', and 'DC Bin Info'")

uploaded_files = st.file_uploader(
    "Drag and drop all 3 files here",
    type=['csv'],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Run Analysis"):

        # --- Run Pipeline ---
        with st.spinner("Running Stage 1: Loading and Identifying Files..."):
            df_cogs, df_history, df_bins = run_stage_1_upload(uploaded_files)

        if df_cogs is not None:
            with st.spinner("Running Stage 2: Cleaning Data..."):
                df_cogs_clean, df_history_clean, df_bin_clean = run_stage_2_cleaning(
                    df_cogs, df_history, df_bins
                )

            if df_cogs_clean is not None:
                with st.spinner("Running Stage 3: Calculating Key Metrics..."):
                    df_summary, cogs_df, bin_df, turnover_df = run_stage_3_analysis(
                        df_cogs_clean, df_history_clean, df_bin_clean
                    )

                if df_summary is not None:
                    with st.spinner("Running Stage 3.2: Generating Visualizations..."):
                        fig, cogs_df, bin_df = run_stage_3_2_graphs(cogs_df, bin_df)

                    with st.spinner("Running Stage 4: Preparing Download..."):
                        excel_bytes = run_stage_4_export(
                            df_summary, cogs_df, bin_df, turnover_df
                        )

                    # --- Display Results ---
                    st.header("Analysis Complete: Results")

                    st.subheader("Key Metrics Summary")
                    st.dataframe(df_summary)

                    st.subheader("Warehouse Analysis Dashboard")
                    if fig:
                        st.pyplot(fig)

                    if excel_bytes:
                        st.download_button(
                            label="Download Master File (Warehouse_Master_File.xlsx)",
                            data=excel_bytes,
                            file_name="Warehouse_Master_File.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
