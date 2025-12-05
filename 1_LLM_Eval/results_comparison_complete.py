"""
Complete code for results_comparison.ipynb
This file contains all the visualization code that should be added to the notebook cells.
Copy sections into notebook cells as indicated.
"""

# ============================================================
# Cell: Binary Comparison Data Preparation
# ============================================================
if COMPARISON_TYPE == "binary":
    data_rows = []
    for entry in filtered_results:
        model_name = entry.get("model_name", "unknown")
        prompt_version = entry.get("prompt_version", "unknown")
        metrics = entry.get("metrics", {})
        date_tested = entry.get("date_tested", "")
        
        row = {
            "model_name": model_name,
            "prompt_version": prompt_version,
            "label": f"{model_name} ({prompt_version})",
            "date_tested": date_tested,
            **metrics
        }
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    print(f"Binary comparison data prepared: {len(df)} entries")
    print(f"\nAvailable metrics: {[col for col in df.columns if col not in ['model_name', 'prompt_version', 'label', 'date_tested']]}")
    print(f"\nData preview:")
    print(df.head())

# ============================================================
# Cell: Binary Comparison Visualizations
# ============================================================
if COMPARISON_TYPE == "binary":
    metric_columns = [col for col in df.columns if col not in ['model_name', 'prompt_version', 'label', 'date_tested']]
    
    if len(metric_columns) > 0:
        n_metrics = len(metric_columns)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metric_columns):
            df_sorted = df.sort_values(metric, ascending=False)
            bars = ax.barh(range(len(df_sorted)), df_sorted[metric].values, color='steelblue')
            ax.set_yticks(range(len(df_sorted)))
            ax.set_yticklabels(df_sorted['label'].values, fontsize=9)
            ax.set_xlabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f"{metric} Comparison", fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)
            
            for i, (bar, val) in enumerate(zip(bars, df_sorted[metric].values)):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        save_path = Path(OUTPUT_DIR) / f"binary_comparison_all_metrics.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot → {save_path}")
        plt.show()
        plt.close()
    
    # Heatmap
    if 'Accuracy' in df.columns or 'F1' in df.columns:
        primary_metric = 'Accuracy' if 'Accuracy' in df.columns else df.columns[4]
        pivot_data = df.pivot_table(values=primary_metric, index='model_name', columns='prompt_version', aggfunc='mean')
        
        if not pivot_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': primary_metric})
            ax.set_title(f"{primary_metric} Heatmap: Model × Prompt Type", fontsize=14, fontweight='bold')
            ax.set_xlabel('Prompt Version', fontsize=12)
            ax.set_ylabel('Model Name', fontsize=12)
            plt.tight_layout()
            save_path = Path(OUTPUT_DIR) / f"binary_heatmap_{primary_metric.lower()}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved plot → {save_path}")
            plt.show()
            plt.close()

# ============================================================
# Cell: Multitype Comparison Data Preparation
# ============================================================
if COMPARISON_TYPE == "multitype":
    data_rows = []
    for entry in filtered_results:
        model_name = entry.get("model", "unknown")
        prompt_type = entry.get("prompt_type", "unknown")
        metrics = entry.get("metrics", {})
        date_tested = entry.get("date_tested", "")
        
        for ipv_type in ["physical", "emotional", "sexual"]:
            if ipv_type in metrics:
                type_metrics = metrics[ipv_type]
                row = {
                    "model_name": model_name,
                    "prompt_type": prompt_type,
                    "label": f"{model_name} ({prompt_type})",
                    "ipv_type": ipv_type,
                    "date_tested": date_tested,
                    **type_metrics
                }
                data_rows.append(row)
    
    df_multitype = pd.DataFrame(data_rows)
    print(f"Multitype comparison data prepared: {len(df_multitype)} entries")
    print(f"\nIPV Types: {df_multitype['ipv_type'].unique().tolist()}")
    print(f"\nAvailable metrics: {[col for col in df_multitype.columns if col not in ['model_name', 'prompt_type', 'label', 'ipv_type', 'date_tested']]}")

# ============================================================
# Cell: Multitype Visualizations - Grouped by IPV Type
# ============================================================
if COMPARISON_TYPE == "multitype":
    metrics_to_plot = [m for m in ['accuracy', 'precision', 'recall', 'f1'] if m in df_multitype.columns]
    
    if len(metrics_to_plot) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for ax, metric in zip(axes[:len(metrics_to_plot)], metrics_to_plot):
            pivot = df_multitype.pivot_table(values=metric, index='label', columns='ipv_type', aggfunc='mean')
            pivot.plot(kind='bar', ax=ax, width=0.8, color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
            ax.set_title(f"{metric.capitalize()} by Model and IPV Type", fontsize=12, fontweight='bold')
            ax.set_xlabel('Model (Prompt Type)', fontsize=10)
            ax.set_ylabel(metric.capitalize(), fontsize=10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.legend(title='IPV Type', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1)
        
        for ax in axes[len(metrics_to_plot):]:
            ax.remove()
        
        plt.tight_layout()
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        save_path = Path(OUTPUT_DIR) / "multitype_comparison_all_metrics.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot → {save_path}")
        plt.show()
        plt.close()

# ============================================================
# Cell: Multitype Visualizations - Per IPV Type
# ============================================================
if COMPARISON_TYPE == "multitype":
    metrics_to_plot = [m for m in ['accuracy', 'precision', 'recall', 'f1'] if m in df_multitype.columns]
    
    for ipv_type in ['physical', 'emotional', 'sexual']:
        if ipv_type not in df_multitype['ipv_type'].values:
            continue
        
        df_type = df_multitype[df_multitype['ipv_type'] == ipv_type]
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics_to_plot):
            df_sorted = df_type.sort_values(metric, ascending=False)
            bars = ax.barh(range(len(df_sorted)), df_sorted[metric].values, color='steelblue')
            ax.set_yticks(range(len(df_sorted)))
            ax.set_yticklabels(df_sorted['label'].values, fontsize=9)
            ax.set_xlabel(metric.capitalize(), fontsize=11, fontweight='bold')
            ax.set_title(f"{ipv_type.capitalize()} Abuse: {metric.capitalize()}", fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)
            
            for bar, val in zip(bars, df_sorted[metric].values):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', ha='left', va='center', fontsize=8)
        
        plt.suptitle(f"{ipv_type.capitalize()} Abuse Performance Comparison", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        save_path = Path(OUTPUT_DIR) / f"multitype_{ipv_type}_comparison.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot → {save_path}")
        plt.show()
        plt.close()

# ============================================================
# Cell: Multitype Heatmap
# ============================================================
if COMPARISON_TYPE == "multitype":
    metrics_to_plot = [m for m in ['accuracy', 'precision', 'recall', 'f1'] if m in df_multitype.columns]
    
    if len(metrics_to_plot) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for ax, metric in zip(axes[:len(metrics_to_plot)], metrics_to_plot):
            pivot = df_multitype.pivot_table(values=metric, index='model_name', columns='ipv_type', aggfunc='mean')
            if not pivot.empty:
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': metric.capitalize()})
                ax.set_title(f"{metric.capitalize()} Heatmap: Model × IPV Type", fontsize=12, fontweight='bold')
                ax.set_xlabel('IPV Type', fontsize=11)
                ax.set_ylabel('Model Name', fontsize=11)
        
        for ax in axes[len(metrics_to_plot):]:
            ax.remove()
        
        plt.tight_layout()
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        save_path = Path(OUTPUT_DIR) / "multitype_heatmap_all_metrics.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot → {save_path}")
        plt.show()
        plt.close()

# ============================================================
# Cell: Best Models Summary Table (Multitype)
# ============================================================
if COMPARISON_TYPE == "multitype":
    print("\n" + "="*80)
    print("BEST MODEL FOR EACH IPV TYPE AND METRIC")
    print("="*80)
    
    metrics_to_analyze = [m for m in ['accuracy', 'precision', 'recall', 'f1'] if m in df_multitype.columns]
    summary_data = []
    
    for ipv_type in ['physical', 'emotional', 'sexual']:
        if ipv_type not in df_multitype['ipv_type'].values:
            continue
        df_type = df_multitype[df_multitype['ipv_type'] == ipv_type]
        for metric in metrics_to_analyze:
            best_idx = df_type[metric].idxmax()
            best_row = df_type.loc[best_idx]
            summary_data.append({
                'IPV Type': ipv_type.capitalize(),
                'Metric': metric.capitalize(),
                'Best Model': best_row['label'],
                'Score': best_row[metric],
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    summary_path = Path(OUTPUT_DIR) / "multitype_best_models_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

