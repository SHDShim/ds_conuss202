Purpose
=======

Assist CONUSS fitting.  This works only for CONUSS ver. 2.0.2.

How to structure your directory
===============================
```
My_Project -- ds_conuss202.py
           |
           -- 15GPa -- Data_file.dat10, Input_in, In_*, ...
           |
           |
           -- 20GPa
           ...
```

How to start
============

0. Run ipython where you have `ds_conuss202.py`.

1. Run
   ```python
   from ds_conuss202 import *
   ```

2. Move to the subdirectory where you have data.

How to setup input files
========================

0. In `in_kfit`, change `exp. data file`.

1. In `in_kref`, change `material data input file`.

2. In `in_kref`, change thickness.

How to conduct CONUSS work
==========================

0. open the input file, data file, and in_kref in a text editor

1. For search, run:
   ```python
    search()
   ```
   Full options: `search(alarm=True, plot_result=True, print_result=True)`

2. For calculate mossbauer spectrum, run:
   ```python
    guess()
   ```
    Full options: `guess(plot_data=True)`

3. For fitting, run:
   ```python
    fit()
   ```
    Full options: `fit(alarm=True, print_result=True, plot_result=True)`

4. For range search, run:
   ```python
   search_range(100, 200, 5)
   ```
   Instruction: `search_range(start, end, num_points)`

Logging information
===================

`search()` and `fit()` make record and backups after execution. They can be found in `backups` subdirectory. There you can find `conuss_log.txt` file. This file has all the record about your fitting and searching. Also, you may find directories named after time.

In case you need to go back to previous fitting, you may use them. In the `backups` subdirectory, you can also find figures of the fitting with names after date and time.
