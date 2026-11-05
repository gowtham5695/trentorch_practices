def selection_sort_passes(nums: list[int]) -> list[list[int]]:
    """Return the list after each pass of selection sort."""
    # TODO: write your solution here
    nums=list(nums)
    passes=[]
    for i in range(len(nums)-1):
        smallest=i
        for j in range(i+1,len(nums)):
            if nums[j]<nums[smallest]:
                smallest=j
        nums[i],nums[smallest]=nums[smallest],nums[i]
        passes.append(list(nums))
    return passes
