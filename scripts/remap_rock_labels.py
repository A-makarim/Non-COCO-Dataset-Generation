"""
Remap rock labels from class 0 to class 1
Run this ONCE before training the 2-class model
"""
from pathlib import Path


def remap_labels(label_dir):
    """Remap all class 0 labels to class 1 in the given directory"""
    label_files = list(Path(label_dir).rglob("*.txt"))
    
    if not label_files:
        print(f"No label files found in {label_dir}")
        return
    
    remapped_count = 0
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            modified = False
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == '0':
                    # Change class 0 to class 1
                    parts[0] = '1'
                    new_lines.append(' '.join(parts))
                    modified = True
                else:
                    new_lines.append(line.strip())
            
            if modified:
                with open(label_file, 'w') as f:
                    f.write('\n'.join(new_lines))
                    if new_lines[-1]:  # Add final newline if content exists
                        f.write('\n')
                remapped_count += 1
        
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    print(f"Remapped {remapped_count} label files in {label_dir}")


def main():
    root = Path(__file__).resolve().parents[1]
    
    # Remap rock labels (data/augmented) from 0 to 1
    rock_dirs = [
        root / "data" / "augmented" / "train" / "labels",
        root / "data" / "augmented" / "val" / "labels",
        root / "data" / "augmented" / "test" / "labels",
    ]
    
    print("Remapping rock labels from class 0 to class 1...")
    for label_dir in rock_dirs:
        if label_dir.exists():
            remap_labels(label_dir)
        else:
            print(f"Directory not found: {label_dir}")
    
    print("\nDone! Rock labels are now class 1, probe labels remain class 0.")
    print("You can now train with data_2class.yaml")


if __name__ == "__main__":
    main()
