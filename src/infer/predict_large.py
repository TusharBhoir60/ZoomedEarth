import torch
import itertools

def define_iteration(dimension: tuple, chunk_size: int, overlap: int = 0):
    dimy, dimx = dimension
    
    # Support smaller dimensions gracefully without returning [(0,0)] if one is large
    # If both are small, just return [(0,0)]
    if dimy <= chunk_size and dimx <= chunk_size:
        return [(0, 0)]
        
    y_step = chunk_size - overlap
    x_step = chunk_size - overlap

    # Ensure we at least have 1 step if a dimension is smaller than chunk_size
    y_range = range(0, dimy, y_step) if dimy > chunk_size else [0]
    x_range = range(0, dimx, x_step) if dimx > chunk_size else [0]

    iterchunks = list(itertools.product(y_range, x_range))
    
    # Fix chunks at the edges
    itercontainer = []
    for index_i, index_j in iterchunks: # index_i is y, index_j is x
        if index_i + chunk_size > dimy:
            index_i = max(dimy - chunk_size, 0)
        if index_j + chunk_size > dimx:
            index_j = max(dimx - chunk_size, 0)
            
        # Deduplicate caused by edge snapping
        pt = (index_i, index_j)
        if pt not in itercontainer:
            itercontainer.append(pt)

    return itercontainer

def predict_large_patched(
    X: torch.Tensor,
    model: torch.nn.Module,
    overlap: int = 32,
) -> torch.Tensor:
    
    H_in, W_in = X.shape[1], X.shape[2]
    
    nruns = define_iteration(
        dimension=(H_in, W_in),
        chunk_size=128,
        overlap=overlap,
    )
    
    output = None
    
    for index, point in enumerate(nruns):        
        # point is (y, x)
        y, x = point[0], point[1]
        
        # Determine actual crop size (might be smaller than 128 if input is smaller)
        crop_h = min(128, H_in - y)
        crop_w = min(128, W_in - x)
        
        Xchunk = X[:, y : y + 128, x : x + 128]
        
        # If input is smaller than 128 in any dimension, pad it to 128 for the model
        pad_h = max(0, 128 - Xchunk.shape[1])
        pad_w = max(0, 128 - Xchunk.shape[2])
        if pad_h > 0 or pad_w > 0:
            Xchunk = torch.nn.functional.pad(Xchunk, (0, pad_w, 0, pad_h), mode="replicate")
            
        result = model(Xchunk[None]).squeeze(0) # (C, 128*res, 128*res)
        
        if index == 0:
            res_n = result.shape[1] // 128
            output = torch.zeros(
                (result.shape[0], H_in * res_n, W_in * res_n),
                dtype=result.dtype,
                device="cpu",
            )
            
        offset_y = y * res_n
        offset_x = x * res_n
        
        # Overlap skipping logic (same conceptually as upstream, but corrected axes)
        # Skip blending on top/left if not the first tile
        skip = overlap * res_n // 2
        
        y_start_in_result = skip if offset_y > 0 else 0
        x_start_in_result = skip if offset_x > 0 else 0
        
        # If this is the last tile, take everything up to the edge
        # Otherwise, skip the bottom/right half of the overlap
        y_end_in_result = 128 * res_n
        if offset_y + 128 * res_n < H_in * res_n:
            y_end_in_result -= skip
            
        x_end_in_result = 128 * res_n
        if offset_x + 128 * res_n < W_in * res_n:
            x_end_in_result -= skip
            
        # Restrict to actual input dimensions if input < 128
        y_end_in_result = min(y_end_in_result, crop_h * res_n)
        x_end_in_result = min(x_end_in_result, crop_w * res_n)
        
        result_cropped = result[:, y_start_in_result:y_end_in_result, x_start_in_result:x_end_in_result]
        
        # Calculate target slice in output
        out_y_start = offset_y + y_start_in_result
        out_x_start = offset_x + x_start_in_result
        out_y_end = out_y_start + result_cropped.shape[1]
        out_x_end = out_x_start + result_cropped.shape[2]
        
        output[:, out_y_start:out_y_end, out_x_start:out_x_end] = result_cropped.detach().cpu()

    return output
