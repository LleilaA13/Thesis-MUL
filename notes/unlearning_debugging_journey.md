# Unlearning Debugging Journey: Mask, Forget Accuracy, and Class Selection

## 1. Initial Setup and Problem
- **Goal:** Unlearn vehicle classes from TinyImageNet using SalUn (saliency-based unlearning)
- **Initial mask used:** `vehicles_forget_indices.pt`
- **Assumption:** This mask targeted vehicle classes (e.g., convertible, moving van, sports car)
- **Evaluation:** Forget accuracy measured on vehicle classes (indices 94, 121, 153 in ImageFolder order)

## 2. Discovery of the Mask Issue
- **Analysis:** The mask was actually targeting unrelated classes:
    - n02788148: bannister, banister, balustrade, handrail
    - n03706229: magnetic compass
    - n04366367: suspension bridge
- **Evidence:** Only 3 classes were marked for forgetting, but they were not vehicles
- **Result:** Forget accuracy on vehicle classes remained high (~73%), so UA was low (~27%)
- **Root cause:** Mismatch between mask class indices and evaluation class indices

## 3. Correction and Verification
- **Created a new mask:** `vehicles_forget_indices_CORRECT.pt` targeting actual vehicle classes present in TinyImageNet:
    - n03100240: convertible (index 94)
    - n03796401: moving van (index 121)
    - n04285008: sports car (index 153)
- **Method:** Used ImageFolder class order to ensure correct mapping
- **Verification:** 1500 samples (3 classes × 500 samples each) now correctly marked for forgetting

## 4. Forget Accuracy and UA Calculation
- **With wrong mask:**
    - Training forget accuracy: 100% (on random classes)
    - Test forget accuracy (on vehicles): ~73%
    - UA: ~27% (misleading, as vehicles were not actually forgotten)
- **With corrected mask:**
    - Expectation: Forget accuracy on vehicle classes should drop significantly after unlearning
    - UA: Should increase, reflecting true unlearning of vehicle classes

## 5. Lessons Learned
- **Always verify mask class mapping** (wnids.txt vs. ImageFolder order)
- **Check actual class descriptions** for all indices in the mask
- **Align mask creation, training, and evaluation** to the same class index system
- **Use interpretable classes (e.g., cats, vehicles)** for feature visualization and thesis clarity

---

**Summary:**
- The initial unlearning experiments were not targeting the intended vehicle classes due to a mask mapping error.
- After correcting the mask, future experiments will accurately measure and demonstrate the effectiveness of class-wise unlearning.
- This debugging process highlights the importance of careful dataset and mask management in machine unlearning research.
