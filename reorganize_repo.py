#!/usr/bin/env python3
"""
Repository Reorganization Script
Reorganizes the machine unlearning repository into a clean, thesis-ready structure
"""

import os
import shutil
from pathlib import Path

def reorganize_repository():
    """Reorganize repository into clean structure"""
    
    base_dir = Path("/media/hdd/usr/leyla/Unlearn-Saliency")
    
    print("🗂️  Starting repository reorganization...")
    print(f"📁 Base directory: {base_dir}")
    
    # Define new structure
    new_structure = {
        "docs": [],
        "experiments": [],
        "analysis": [],
        "utils": [],
        "config": [],
        "data": ["datasets", "masks", "models", "results", "labels", "visuals"],
    }
    
    # Create new directories
    for dir_name in new_structure.keys():
        new_dir = base_dir / dir_name
        new_dir.mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}/")
    
    # Create data subdirectories
    for subdir in new_structure["data"]:
        (base_dir / "data" / subdir).mkdir(exist_ok=True, parents=True)
        print(f"✅ Created directory: data/{subdir}/")
    
    # Define file movements
    movements = [
        # Documentation
        ("transfer_learning_analysis.md", "docs/"),
        ("thesis_visualization_guide.md", "docs/"),
        ("TODO.md", "docs/"),
        ("to_do.md", "docs/"),
        ("todo2.md", "docs/"),
        ("salUN.md", "docs/"),
        
        # Experiments
        ("resnet50_unlearn.py", "experiments/"),
        ("inceptionv3_unlearn.py", "experiments/"),
        ("create_vehicle_forget_mask.py", "experiments/"),
        
        # Analysis
        ("generate_thesis_plots.py", "analysis/"),
        ("generate_feature_plots.py", "analysis/"),
        ("generate_evaluation_plots.py", "analysis/"),
        ("run_all_plots.py", "analysis/"),
        ("inspect_mask.py", "analysis/"),
        
        # Utils
        ("test_tinyimagenet.py", "utils/"),
        ("debug_tinyimagenet.py", "utils/"),
        
        # Config
        ("environment.yml", "config/"),
        
        # Data files (move to appropriate data subdirectories)
        ("cat_forget_indices.pt", "data/labels/"),
        ("vehicles_forget_indices.pt", "data/labels/"),
        ("marked_labels.pt", "data/labels/"),
    ]
    
    # Move existing directories
    directory_movements = [
        ("datasets", "data/datasets"),
        ("masks", "data/masks"),
        ("models", "data/models"),
        ("results", "data/results"),
        ("labels", "data/labels"),
        ("visuals", "data/visuals"),
        ("old_scripts", "utils/old_scripts"),
    ]
    
    # Execute file movements
    print("\n📦 Moving files...")
    for src, dst in movements:
        src_path = base_dir / src
        dst_path = base_dir / dst / src
        
        if src_path.exists():
            try:
                # Create destination directory if it doesn't exist
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_path), str(dst_path))
                print(f"📋 Moved: {src} → {dst}")
            except Exception as e:
                print(f"❌ Failed to move {src}: {e}")
        else:
            print(f"⚠️  File not found: {src}")
    
    # Execute directory movements
    print("\n📂 Moving directories...")
    for src, dst in directory_movements:
        src_path = base_dir / src
        dst_path = base_dir / dst
        
        if src_path.exists() and src_path.is_dir():
            try:
                # Create parent directory if needed
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                # If destination exists, merge contents
                if dst_path.exists():
                    # Move contents instead of directory
                    for item in src_path.iterdir():
                        item_dst = dst_path / item.name
                        if item_dst.exists():
                            if item.is_dir():
                                shutil.rmtree(str(item_dst))
                            else:
                                item_dst.unlink()
                        shutil.move(str(item), str(item_dst))
                    src_path.rmdir()
                else:
                    shutil.move(str(src_path), str(dst_path))
                
                print(f"📁 Moved: {src}/ → {dst}/")
            except Exception as e:
                print(f"❌ Failed to move directory {src}: {e}")
        else:
            print(f"⚠️  Directory not found: {src}")
    
    # Create thesis_figures if it doesn't exist
    thesis_figures = base_dir / "thesis_figures"
    if not thesis_figures.exists():
        thesis_figures.mkdir()
        print("📊 Created: thesis_figures/")
    
    # Create important subdirectories
    important_dirs = [
        "thesis_figures/feature_visualizations",
        "thesis_figures/evaluation_metrics", 
        "data/checkpoints",
        "reports",
    ]
    
    for dir_path in important_dirs:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"📋 Created: {dir_path}/")
    
    print("\n✨ Repository reorganization complete!")
    
    # Generate new directory tree
    print("\n📊 New repository structure:")
    print_directory_tree(base_dir, max_depth=3)
    
    return True

def print_directory_tree(path, prefix="", max_depth=3, current_depth=0):
    """Print a directory tree structure"""
    if current_depth > max_depth:
        return
    
    path = Path(path)
    if not path.is_dir():
        return
    
    items = sorted([p for p in path.iterdir() if not p.name.startswith('.')])
    dirs = [p for p in items if p.is_dir()]
    files = [p for p in items if p.is_file()]
    
    # Print directories first
    for i, directory in enumerate(dirs):
        is_last_dir = (i == len(dirs) - 1) and len(files) == 0
        current_prefix = "└── " if is_last_dir else "├── "
        print(f"{prefix}{current_prefix}📁 {directory.name}/")
        
        # Recursive call for subdirectories
        extension = "    " if is_last_dir else "│   "
        print_directory_tree(directory, prefix + extension, max_depth, current_depth + 1)
    
    # Print files (only show a few key ones to avoid clutter)
    important_files = ["README.md", "LICENSE", ".gitignore", "environment.yml"]
    display_files = [f for f in files if f.name in important_files or current_depth == 0]
    
    for i, file in enumerate(display_files[:5]):  # Limit to 5 files
        is_last = i == len(display_files) - 1
        current_prefix = "└── " if is_last else "├── "
        file_icon = "📄" if file.suffix in ['.md', '.txt'] else "📋" if file.suffix in ['.py'] else "📦"
        print(f"{prefix}{current_prefix}{file_icon} {file.name}")
    
    if len(files) > 5 and current_depth <= 1:
        print(f"{prefix}    ... and {len(files) - 5} more files")

def create_project_summary():
    """Create a project summary file"""
    base_dir = Path("/media/hdd/usr/leyla/Unlearn-Saliency")
    
    summary_content = """# Project Summary

## Repository Organization Status

✅ **Completed**: Repository restructured into logical components
✅ **Documentation**: Moved to docs/ directory  
✅ **Experiments**: Centralized in experiments/ directory
✅ **Analysis**: Visualization scripts in analysis/ directory
✅ **Data Management**: All data files organized under data/
✅ **Configuration**: Environment files in config/

## Next Steps

1. **Update README**: Replace current README.md with README_new.md
2. **Update Imports**: Some scripts may need import path updates
3. **Documentation**: Update any hardcoded paths in scripts
4. **Git Management**: Update .gitignore for new structure

## File Movements Summary

### Core Scripts
- resnet50_unlearn.py → experiments/
- analysis scripts → analysis/
- documentation → docs/

### Data Organization
- All datasets → data/datasets/
- Model checkpoints → data/models/
- Results → data/results/
- Masks → data/masks/

### Benefits
- 📁 Clear separation of concerns
- 🔍 Easy navigation for thesis reviewers
- 📚 Professional documentation structure
- 🛠️ Simplified development workflow
"""
    
    summary_path = base_dir / "REORGANIZATION_SUMMARY.md"
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    
    print(f"📋 Created reorganization summary: {summary_path}")

if __name__ == "__main__":
    # Run reorganization
    success = reorganize_repository()
    
    if success:
        create_project_summary()
        print("\n🎉 Repository is now organized and thesis-ready!")
        print("\n📋 Next steps:")
        print("1. Review the new structure")
        print("2. Replace README.md with README_new.md")
        print("3. Update any hardcoded paths in scripts")
        print("4. Test that main experiments still work")
    else:
        print("❌ Reorganization failed. Please check errors above.")