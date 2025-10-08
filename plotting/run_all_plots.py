#!/usr/bin/env python3
"""
Master Script to Generate All Thesis Visualizations
Runs all plotting scripts and creates a comprehensive figure collection
"""

import os
import sys
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    """Run all plotting scripts and generate thesis figures"""
    
    base_dir = Path("/media/hdd/usr/leyla/Unlearn-Saliency")
    scripts_dir = base_dir
    
    print("🎨 Starting comprehensive thesis visualization generation...")
    print("=" * 60)
    
    # List of plotting scripts to run
    plotting_scripts = [
        "generate_thesis_plots.py",
        "generate_feature_plots.py", 
        "generate_evaluation_plots.py"
    ]
    
    # Check if scripts exist
    missing_scripts = []
    for script in plotting_scripts:
        script_path = scripts_dir / script
        if not script_path.exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"❌ Missing scripts: {missing_scripts}")
        return
    
    # Run each plotting script
    for script in plotting_scripts:
        print(f"\n🔄 Running {script}...")
        try:
            script_path = scripts_dir / script
            result = subprocess.run([sys.executable, str(script_path)], 
                                  capture_output=True, text=True, cwd=str(scripts_dir))
            
            if result.returncode == 0:
                print(f"✅ {script} completed successfully")
                # Print any output
                if result.stdout:
                    print(result.stdout)
            else:
                print(f"❌ {script} failed with error:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Error running {script}: {e}")
    
    # Create index of all generated figures
    create_figure_index()
    
    print("\n" + "=" * 60)
    print("🎉 Thesis visualization generation complete!")
    print(f"📁 Check the thesis_figures/ directory for all outputs")

def create_figure_index():
    """Create an HTML index of all generated figures"""
    
    base_dir = Path("/media/hdd/usr/leyla/Unlearn-Saliency")
    figures_dir = base_dir / "thesis_figures"
    
    if not figures_dir.exists():
        print("⚠️  No thesis_figures directory found")
        return
    
    # Find all PNG files
    png_files = list(figures_dir.rglob("*.png"))
    
    if not png_files:
        print("⚠️  No PNG files found in thesis_figures")
        return
    
    # Create HTML index
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Thesis Figures Index</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .figure {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; }}
        .figure img {{ max-width: 800px; height: auto; }}
        .figure h3 {{ color: #333; }}
        .category {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Machine Unlearning Thesis: Figure Collection</h1>
    <p>Generated on: {Path().cwd()}</p>
    <p>Total figures: {len(png_files)}</p>
    
    <h2>Figure Categories</h2>
"""
    
    # Group figures by subdirectory
    categories = {}
    for png_file in png_files:
        relative_path = png_file.relative_to(figures_dir)
        category = relative_path.parent.name if relative_path.parent != Path(".") else "Main"
        
        if category not in categories:
            categories[category] = []
        categories[category].append(png_file)
    
    # Generate HTML for each category
    for category, files in categories.items():
        html_content += f"""
    <div class="category">
        <h2>{category.replace('_', ' ').title()}</h2>
        <p>{len(files)} figures</p>
"""
        
        for png_file in sorted(files):
            relative_path = png_file.relative_to(figures_dir)
            figure_name = png_file.stem.replace('_', ' ').title()
            
            html_content += f"""
        <div class="figure">
            <h3>{figure_name}</h3>
            <img src="{relative_path}" alt="{figure_name}">
            <p><strong>File:</strong> {relative_path}</p>
        </div>
"""
        
        html_content += "    </div>\n"
    
    html_content += """
</body>
</html>
"""
    
    # Save HTML file
    index_path = figures_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Figure index created: {index_path}")
    
    # Also create a simple text summary
    summary_path = figures_dir / "figures_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("THESIS FIGURES SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for category, files in categories.items():
            f.write(f"{category.upper()}:\n")
            f.write("-" * 20 + "\n")
            for png_file in sorted(files):
                f.write(f"  • {png_file.name}\n")
            f.write(f"  Total: {len(files)} figures\n\n")
        
        f.write(f"TOTAL FIGURES: {len(png_files)}\n")
    
    print(f"📄 Text summary created: {summary_path}")

if __name__ == "__main__":
    main()