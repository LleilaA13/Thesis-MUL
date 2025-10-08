"""
Centralized configuration for unlearning experiments on TinyImageNet-200
"""

# TinyImageNet-200 class configurations
TINYIMAGENET_CLASSES = {
    'dogs': {
        'wnids': ['n02085620', 'n02094433', 'n02099601', 'n02099712', 'n02106662', 'n02113799'],
        'indices': [182, 135, 78, 39, 11, 194],  # Corresponding indices in wnids.txt
        'names': ['Chihuahua', 'Yorkshire Terrier', 'Golden Retriever', 'Labrador Retriever', 'German Shepherd', 'Standard Poodle'],
        'description': 'Dog breeds for canine class forgetting'
    },
    'cats': {
        'wnids': ['n02124075', 'n02123045', 'n02125311', 'n02123394'],
        'indices': [0, 66, 102, 131],  # Corresponding indices in wnids.txt
        'names': ['Egyptian Cat', 'Tabby Cat', 'Cougar/Mountain Lion', 'Persian Cat'],
        'description': 'Feline classes for cat forgetting'
    },
    'vehicles': {
        'wnids': ['n02814533', 'n02963159', 'n03393912', 'n03796401', 'n03977966', 'n04146614', 'n04285008', 'n04487081'],
        'indices': [147, 31, 52, 64, 90, 15, 117, 152],  # Corresponding indices in wnids.txt  
        'names': ['Beach Wagon', 'Car', 'Golf Cart', 'Go-kart', 'Police Van', 'School Bus', 'Sports Car', 'Taxi'],
        'description': 'Vehicle classes for transportation forgetting'
    }
}

def get_forget_class_config(forget_type):
    """
    Get configuration for a specific forget class type
    
    Args:
        forget_type (str): 'dogs', 'vehicles', or custom class name
        
    Returns:
        dict: Configuration with wnids, indices, names, and description
    """
    if forget_type in TINYIMAGENET_CLASSES:
        return TINYIMAGENET_CLASSES[forget_type]
    else:
        raise ValueError(f"Unknown forget type: {forget_type}. Available: {list(TINYIMAGENET_CLASSES.keys())}")

def get_forget_indices_for_dataset(forget_type, dataset_type='train'):
    """
    Get sample indices for forgetting based on class type
    
    Args:
        forget_type (str): 'dogs', 'vehicles', etc.
        dataset_type (str): 'train' or 'val'
        
    Returns:
        list: Indices of samples to forget
    """
    config = get_forget_class_config(forget_type)
    class_indices = config['indices']
    
    # For TinyImageNet, each class has 500 training samples (0-499) and 50 val samples
    # Classes are ordered sequentially in the dataset
    forget_indices = []
    
    for class_idx in class_indices:
        if dataset_type == 'train':
            # Each class has 500 training samples
            start_idx = class_idx * 500
            end_idx = start_idx + 500
        elif dataset_type == 'val':
            # Each class has 50 validation samples  
            start_idx = class_idx * 50
            end_idx = start_idx + 50
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
            
        forget_indices.extend(range(start_idx, end_idx))
    
    return sorted(forget_indices)

def create_forget_mask(forget_type, total_samples=100000, dataset_type='train'):
    """
    Create boolean mask for forgetting
    
    Args:
        forget_type (str): 'dogs', 'vehicles', etc.
        total_samples (int): Total number of samples in dataset
        dataset_type (str): 'train' or 'val'
        
    Returns:
        torch.Tensor: Boolean mask where True = forget, False = retain
    """
    import torch
    
    forget_indices = get_forget_indices_for_dataset(forget_type, dataset_type)
    mask = torch.zeros(total_samples, dtype=torch.bool)
    mask[forget_indices] = True
    
    return mask

def print_config_summary():
    """Print summary of all available configurations"""
    print("=== TinyImageNet-200 Unlearning Configurations ===")
    for forget_type, config in TINYIMAGENET_CLASSES.items():
        print(f"\n{forget_type.upper()}:")
        print(f"  Classes: {len(config['wnids'])}")
        print(f"  WNIDs: {config['wnids']}")
        print(f"  Indices: {config['indices']}")
        print(f"  Names: {config['names']}")
        print(f"  Description: {config['description']}")
        
        # Calculate expected sample counts
        num_classes = len(config['indices'])
        train_samples = num_classes * 500
        val_samples = num_classes * 50
        print(f"  Training samples to forget: {train_samples}")
        print(f"  Validation samples to forget: {val_samples}")

if __name__ == "__main__":
    print_config_summary()