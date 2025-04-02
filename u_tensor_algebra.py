import numpy as np
import opt_einsum as oe
import jax.numpy as jnp
from itertools import combinations

# Automated Symbolic Derivation of Unbiased Linear Tensor Algebra

# Author: Chanwoo Chun
# Date: 2025

# Usage instruction:
# Call get_unbiased_einsums to derive the expression of the unbiased estimator
# Then call compute_estimate to apply the estimator on a real dataset

def expand_indicator_pairs(pairs_of_indices):
    """
    Given a list of pairs_of_indices = [(i1, i2), (i3, i4), ...]
    for each pair we have factor (1 - delta_{i1, i2}),
    expand the product into 2^len(pairs) terms.

    Returns a list of (coefficient, merges) 
    where 'merges' is a list of index-equalities 
    that come from including the delta in that term.
    
    Example: for 2 pairs (i1,i2), (a1,a2),
       we get:
         +1, []                       (no merges)
         -1, [(i1, i2)]               (merging i1 and i2)
         -1, [(a1, a2)]
         +1, [(i1, i2), (a1, a2)]
    """
    # We'll do a binary choice for each pair: either we pick "+1" (the '1' part)
    # or we pick "-1" and the merges (the 'delta' part).
    # This yields 2^m expansions if we have m pairs.
    from itertools import product

    terms = []
    m = len(pairs_of_indices)
    for subset_mask in product([False, True], repeat=m):
        # subset_mask[i] = True means we pick the delta in the i-th pair
        # subset_mask[i] = False means we pick the '1'
        merges = []
        coeff = 1
        for pick_delta, (x, y) in zip(subset_mask, pairs_of_indices):
            if pick_delta:
                # picking delta_{x,y} => coefficient is -1
                # and merges = (x,y)
                coeff *= -1
                merges.append((x, y))
            else:
                # picking '1' => do nothing to merges, coeff stays
                pass
        terms.append( (coeff, merges) )
    return terms


def merge_indices_in_einsum(merges, all_indices):
    """
    Given a list of merges = [(i1,i2), (a1,a2), ...],
    we want to unify those pairs of indices so they become a single dimension
    in an einsum expression.

    all_indices: a dictionary like {'X':('i1','a1'), 'Y':('i2','a2'), ...}
                 telling which symbolic indices each factor uses.
    factor_shapes: a dictionary mapping factor names to actual numpy arrays.

    We'll produce:
      1) A new set of factor->index pattern after merges
      2) an einsum string that sums over all resulting indices
      3) a list of the actual arrays for input to opt_einsum
    """

    # Step 1: we need to keep track of equivalence classes of indices
    # e.g. if (i1,i2) is in merges, i1 and i2 are the same dimension
    # We'll do a naive "union-find" approach or just a dictionary.
    parent = {}
    def find(x):
        # find representative
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        rx, ry = find(x), find(y)
        parent[ry] = rx

    # Initialize parent
    unique_indices = set()
    for factor, idxs in all_indices.items():
        for idx in idxs:
            unique_indices.add(idx)
    for idx in unique_indices:
        parent[idx] = idx

    # unify merges
    for (x, y) in merges:
        union(x, y)

    # Step 2: rewrite the factor index patterns
    # e.g. if factor X had (i1,a1), and merges has (i1,i2), 
    # and i2 was also found to be the same as i3, etc...
    # we produce a new index label = find(old_index).
    new_factor_idx = {}
    for factor, idxs in all_indices.items():
        new_idx_tuple = tuple(find(ix) for ix in idxs)
        new_factor_idx[factor] = new_idx_tuple
        
    # Step 3: Build the einsum string
    # We gather all unique "representative" indices from new_factor_idx.
    # Then the summation is over all repeated indices among factors.
    # A straightforward approach: 
    #   - assign each representative index to a letter (like 'a','b','c'...) 
    #   - build an einsum subscript for each factor
    #   - join them with ',' -> '->'
    #   - call opt_einsum
    # But we also want to sum out everything (to get a scalar). 
    #   => '->' with no indices on the right side in Einstein notation.
    # create a stable mapping from representative index to single-letter label
    sorted_reps = sorted(set(find(ix) for ix in unique_indices))

    label_map = {}
    import string
    # For safety if we have many indices, we might need more than 26 letters or 
    # use something like an extended scheme, but let's assume fewer than 26 for demo.
    letters = list(string.ascii_lowercase)  
    for i, rep in enumerate(sorted_reps):
        label_map[rep] = letters[i]

    # Now build for each factor something like 'ab'
    factor_subscripts = []
    factor_names = []
    for factor, new_idxs in new_factor_idx.items():
        factor_sub = ''.join(label_map[idx_rep] for idx_rep in new_idxs)
        factor_subscripts.append(factor_sub)
        factor_names.append(factor)

    # Final einsum subscript:
    #   'ab,ac,bc,...->'  (no output indices => fully summed)
    full_sub = ','.join(factor_subscripts) + '->'

    return full_sub, factor_names


def index_branching_for_centering(all_indices,centerings,dist_groups):
    """
    Centering operation requires branching the indices. 
    
    Input:
    all_indices: The original set of indices
    centerings: Centering instruction
    dist_groups: Specify which groups of indices which be distinct
    
    Output:
    deltas: List of centraction pairs (the pair being original name and the new name)
    all_indices_new: Updated version of all_indices
    dist_groups_new: Updated version of dist_groups
    all_symbols: All unique new index names
    """
    
    dist_groups_new = dist_groups.copy()
    
    idx_dic={}
    all_indices_new={}
    deltas=[]
    all_symbols=set()
    for fa_name,indices,centers in zip(all_indices.keys(),all_indices.values(),centerings.values()):
        #print(indices)
        new_indices=[]
        for idx,center in zip(indices,centers):
            if center=='c':
                if idx in idx_dic:
                    count=int(idx_dic[idx][-1][-1])
                    #print(idx+'_'+str(count+1)
                    new_name=idx+'_'+str(count+1)
                    idx_dic[idx]=idx_dic[idx]+[new_name]
                else:
                    new_name=idx+'_1'
                    idx_dic[idx]=[new_name]
                new_indices.append(new_name)
                deltas.append((idx,new_name))
    
                for j,dist_group in enumerate(dist_groups_new):
                    if idx in dist_group:
                        dist_group_new = dist_group + (new_name,)
                        dist_groups_new[j]=dist_group_new

                all_symbols.add(new_name)
            else:
                new_indices.append(idx)
                
            all_symbols.add(idx)
            
            
        new_indices=tuple(new_indices)
        all_indices_new[fa_name]=new_indices
    
    return deltas,all_indices_new,dist_groups_new,all_symbols


def expand_centering_delta_pairs(pairs_of_indices):
    """
    Expand the (1-delta) polynomial obtained from the centering operation
    """
    from itertools import product

    terms = []
    m = len(pairs_of_indices)
    for subset_mask in product([False, True], repeat=m):
        # subset_mask[i] = True means we pick the delta in the i-th pair
        # subset_mask[i] = False means we pick the '1'
        merges = []
        coeff = 1
        for pick_delta, (x, y) in zip(subset_mask, pairs_of_indices):
            if pick_delta:
                # picking delta_{x,y} => coefficient is -1
                # and merges = (x,y)
                merges.append((x, y))
            else:
                # picking '1' => do nothing to merges, coeff stays
                coeff *= -1
                pass
        terms.append( (coeff, merges) )
    return terms

def center_contract_and_rename(deltas, all_indices_new, dist_groups_new, all_symbols):
    """
    This function takes the output from "index_branching_for_centering".
    This function contracts indices for each term in the summation expansion
    Returns:
    coeff_list: Coefficients for each term in the sum expansion
    dist_groups_list: Names of distinct groups of indices 
    all_indices_list: Names of all indices
    """
    
    center_coeffs_merges = expand_centering_delta_pairs(deltas)
    #print(center_coeffs_merges)
    
    coeff_list = []
    dist_groups_list = []
    all_indices_list = []
    
    # contract indices and rename
    for (coeff,merges) in center_coeffs_merges:
    
        parent = {}
        def find(x):
            # find representative
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            rx, ry = find(x), find(y)
            parent[ry] = rx
        
        # Initialize parent
        unique_indices = list(all_symbols)
        
        for idx in unique_indices:
            parent[idx] = idx

        # unify merges
        for (x, y) in merges:
            union(x, y)
                
        #all_indices_new
        #dpick relevant ones from ist_groups_new
        all_indices_contracted={}
        all_symbols_contracted=set()
        for fa_name,indices in zip(all_indices_new.keys(),all_indices_new.values()):
            indices_new = []
            for idx in indices:
                new_idx = find(idx)
                all_symbols_contracted.add(new_idx)
                indices_new.append(new_idx)
            all_indices_contracted[fa_name]=tuple(indices_new)
        all_symbols_contracted=sorted(all_symbols_contracted)
        

        dist_groups_contracted=[]
        for dgn in dist_groups_new:
            dist_groups_contracted.append( tuple(sorted( set(dgn).intersection(set(all_symbols_contracted) ))))
 
        #all_indices_contracted: the first functional form
        #dist_groups_contracted: the first functional distinct indices
    
        ## From here on is about renaming
        all_symbols_dict={}
        #for i,symb in enumerate(all_symbols_contracted):
        #    all_symbols_dict[symb]=str(i)
        countv=0
        for indices in all_indices_contracted.values():
            for idx in indices:
                if idx not in all_symbols_dict:
                    all_symbols_dict[idx]=str(countv)
                    countv=countv+1

        all_indices_contracted_renamed={}
        for faname,indices in zip(all_indices_contracted.keys(),all_indices_contracted.values()):
            indices_renamed=[]
            for idx in indices:
                indices_renamed.append(all_symbols_dict[idx])
            all_indices_contracted_renamed[faname] = tuple(indices_renamed)
        
        dist_groups_contracted_renamed=[]
        for dgn in dist_groups_contracted:
            dgn_new=()
            for idx in dgn:
                dgn_new+=(all_symbols_dict[idx],)
            dist_groups_contracted_renamed.append(tuple(sorted(dgn_new)))
            #print(dgn_new)
        #print('coeff:', coeff)
        #print('dist_groups_contracted_renamed: ',dist_groups_contracted_renamed)
        #print('all_indices_contracted_renamed: ',all_indices_contracted_renamed)
        
        coeff_list.append(coeff)
        dist_groups_list.append(dist_groups_contracted_renamed)
        all_indices_list.append(all_indices_contracted_renamed)
    return coeff_list,dist_groups_list,all_indices_list


def simplify_and_get_final_formulas(coeff_list,dist_groups_list,all_indices_list):
    """
    The final simplification of the symbolic expression based on symmetries in the data.
    Takes the output from "center_contract_and_rename" as input.
    It returns Distincts_A, All_IDX_A, which will be used 
    """
    # simplify
    unique_dict = {}
    for coeff, dist_groups, all_indices in zip(coeff_list, dist_groups_list, all_indices_list):
        # Step 1: Make a hashable version of dist_groups
        #    If dist_groups is something like [(0,2,3,5,6,8), (1,4,7)],
        #    we can turn each sub-list into a tuple, and then the entire list-of-lists into a tuple of tuples.
        #    For example:
        #
        #       hashable_dist_groups = tuple(tuple(g) for g in dist_groups)
        #
        # Step 2: Make a hashable version of all_indices
        #    If it's a dict, we can do sorted(...) by keys, then turn to a tuple of (key, val).
        #    For example:
        #
        #       hashable_all_indices = tuple(sorted(all_indices.items()))
        #
        # This ensures that two identical structures produce the same key.
    
        hashable_dist_groups = tuple(tuple(g) for g in dist_groups)  
        hashable_all_indices = tuple(sorted(all_indices.items()))     
    
        # Combine them into a single key
        combination_key = (hashable_dist_groups, hashable_all_indices)
    
        # Step 3: Accumulate (sum up) the coeff
        if combination_key not in unique_dict:
            unique_dict[combination_key] = coeff
        else:
            unique_dict[combination_key] += coeff
    
    # At this point, unique_dict has entries like:
    #    {
    #       ( ( (0,2,3,5,6,8),(1,4,7) ), ( ('X1',(0,1)),('X2',(2,1)),... ) ): <sum_of_coeffs>,
    #       ...
    #    }
    
    Coe_A=[]
    N_A=[]
    Distincts_A=[]
    All_IDX_A=[]
    for combo, coeff_sum in unique_dict.items():
        #unique_combinations_list.append(combo)    # combo is ((dist_groups), (sorted_all_indices))
    
        coe=coeff_sum
        dins=list(combo[0])
        aid_in_tuple_format=combo[1]
    
        # we need to convert the aid_in_tuple_format into dictionary 
        # (it was originally in tuple, since we wanted to hash it earlier for simplifying the terms) 
        aid_in_dic={} 
        N_names={}
        aidv=set()
        for adx in aid_in_tuple_format:
            vname=adx[0]
            indices_tuple=adx[1]
            aid_in_dic[vname]=indices_tuple # work needed to convert to dictionary is done here
    
            for i,idx in enumerate(indices_tuple):
                if i==0:
                    Nn='P'
                elif i==1:
                    Nn='Q'
                if idx not in N_names:
                    N_names[idx] = Nn+'_'+vname
                aidv.add(idx)
                    
        #### Now compute the correct number of summands for the coefficients for the averaging
        N_coeffs=[]
        dins_concatenate=()
        for din in dins:
            total_din=len(din) # total count of distinct indices in a given set of distinct indices
            first_one=din[0]
            Nv=N_names[first_one]
            for c in range(total_din):
                N_coeffs.append(Nv+'-'+str(c))
            dins_concatenate+=din
        
        # Now that we covered the distinct indices, we need to take care of the rest
        
        rest_idcs=aidv-set(dins_concatenate)
        for idx in rest_idcs:
            Nv=N_names[idx]
            N_coeffs.append(Nv+'-'+str(0))
    
        print('Distinct indices: ', dins)
        print('Add or Subtract: ', coe)
        print('Formula: ', aid_in_dic)
        print('')
        Coe_A.append(coe)
        N_A.append(N_coeffs)
        Distincts_A.append(dins)
        All_IDX_A.append(aid_in_dic)
        
    return Coe_A, N_A, Distincts_A, All_IDX_A

def get_all_einsums(Distincts_A, All_IDX_A):
    """
    Compute unbiased estimator for all summations, in the Einsum syntax. Each summation has multiple Einsums and we have a single coefficient for each Einsum.
    """
    contractions=[]
    einsum_A=[]
    for dist_grps,AIdcs in zip(Distincts_A,All_IDX_A):
        pairs_of_id=()
        for group in dist_grps:
            pairs_of_id+=tuple(combinations(group,2))
        pairs_of_id=list(pairs_of_id)

        contractions = expand_indicator_pairs(pairs_of_id)
        #contractions.append(terms)

        einsum_data=[]
        for (coeff, merges) in contractions:
        # merges might be [] or [(i1,i2)] or [(a1,a2)] or both
            subscript, factor_names = merge_indices_in_einsum(merges, AIdcs)
            einsum_data.append([coeff,subscript,factor_names])
        einsum_A.append(einsum_data)
    return einsum_A


def get_unbiased_einsums(all_indices,centerings,dist_groups):
    """
    This is the parent function that takes in the instruction and returns the final expression
    """
    ### CENTERING ###
    deltas,all_indices_new,dist_groups_new,all_symbols = index_branching_for_centering(all_indices,centerings,dist_groups)
    coeff_list,dist_groups_list,all_indices_list=center_contract_and_rename(deltas, all_indices_new, dist_groups_new, all_symbols)
    Coe_A, N_A, Distincts_A, All_IDX_A = simplify_and_get_final_formulas(coeff_list,dist_groups_list,all_indices_list)
    # All_IDX_A is a list of M summations formulas, Coe_A is the list of cofficients for each formula, and N_A is the symbolic counts of the summands
    #################
    ### Compute unbiased estimator for all summations, in the Einsum syntax. Each summation has multiple Einsums and we have a single coefficient for each Einsum.
    einsum_A = get_all_einsums(Distincts_A, All_IDX_A)
    #################
    return (einsum_A, Coe_A, N_A)

###################### HANDLING ACTUAL DATA #####################################
def compute_estimate(factor_data, estimator_formula):
    """
    einsum_A in estimator_formula is the expression derived by get_all_einsums (or the output of get_unbiased_einsums)
    It takes in the real data and computes the estimation based on einsum_A instruction
    """
    einsum_A, Coe_A, N_A = estimator_formula
    
    #get shapes
    factor_shapes={}
    for fn in factor_data:
        factor_shapes[fn]=np.shape(factor_data[fn])
    print(factor_shapes)
    print('')
    
    count_nterms=0
    final=0
    for coe, N_coeffs, einsum_data in zip(Coe_A,N_A,einsum_A):
        
        fvals_raw=[]
        fvals=[]
        for N_coeff in N_coeffs:
            split1=N_coeff.split("_", 1)
            
            N_name=split1[0]
            if N_name=='P':
                axis_idx=0
            elif N_name=='Q':
                axis_idx=1
            split2=split1[1].split("-")
            fct_name=split2[0]
            subs=int(split2[1])
            #print(axis_idx, fct_name, subs)
            fvals_raw.append(factor_shapes[fct_name][axis_idx])
            fvals.append(factor_shapes[fct_name][axis_idx]-subs)
        #print(fvals_raw)
        #print(fvals)
        logprod=np.sum([np.log(fval) for fval in fvals])
        #print(AIdcs)
        #print(N_coeffs)
        
        total_sum = 0.0
        vals=[]
        for einsum_datum in einsum_data:
            coeff = einsum_datum[0]
            subscript = einsum_datum[1]
            factor_names = einsum_datum[2]
            
            normalize=np.exp(-logprod/len(factor_names))
            fac_arrays = [factor_data[fc]*normalize for fc in factor_names]
            
            val = oe.contract(subscript, *fac_arrays)
            # Now multiply by the coefficient
            total_sum += coeff * val
            vals.append(float(val))
    
            count_nterms+=1
        #print(total_sum)
        final+=coe*total_sum
        #print(AIdcs)
        print(coe*total_sum)
    print('')
    print('Total number of terms: ', count_nterms)
    print('Estimate: ', final)
    
    return final
    