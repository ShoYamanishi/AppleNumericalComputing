import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import sys

BASE_DIR = 'doc/'

def parse_commandline():

    arg_parser = argparse.ArgumentParser( description = 'generate plots from the log output' )
    arg_parser.add_argument( '-logfile',   nargs = '?', default='make_log.txt',   help = 'log file' )
    arg_parser.add_argument( '-specfile',  nargs = '?', default='plot_spec.json', help = 'plot specification file' )       
    arg_parser.add_argument( '-plot_charts', action = "store_true", help = 'generates plots per spec' )
    arg_parser.add_argument( '-show_impl', action = "store_true", help = 'shows all the implementation types' )
    return arg_parser


def get_unique_values_for_column(df, col):
    L = []
    for v in df[col].unique():
        L.append(v)
    return L


def get_implementation_combinations(df):

    implementation_types   = get_unique_values_for_column( df, 'implementation type'  )
    loop_unrolling_factors = get_unique_values_for_column( df, 'loop unrolling factor')
    num_cpu_threads        = get_unique_values_for_column( df, 'num CPU threads'      )

    L = []
    for t in implementation_types:

        if t == 'METAL':       
            metal_implementations = get_unique_values_for_column( df, 'metal implementation type' )
            metal_implementations.remove('NOT_APPLICABLE')
            for mi in metal_implementations:
                df_filtered = df[ (df['implementation type'] == t ) & (df['metal implementation type'] == mi) ]
                if df_filtered.shape[0] > 0:
                    L.append( t + " " + mi + " 0 0" )
        else:
            for f in loop_unrolling_factors:
                for n in num_cpu_threads:
                    df_filtered = df[ (df['implementation type'] == t ) & (df['loop unrolling factor'] == f) & (df['num CPU threads'] == n) ]
                    if df_filtered.shape[0] > 0:
                        L.append( t + " " + str(f) + " " + str(n) )
    return L


def get_best_times_for_impl( vector_length, mean_times ):

    new_lenghts = []
    new_times   = []
    prev_best = 1.0e+20
    prev_length = 0
    for i in range( 0, len(vector_length) ):
        v = vector_length[i]
        t = mean_times   [i]
        if prev_length == v:
            if prev_best > t:
                prev_best = t
        else:
            if prev_length != 0:
                new_lenghts.append( prev_length )
                new_times.append  ( prev_best   )
            prev_length = v
            prev_best   = t

    new_lenghts.append( prev_length )
    new_times.append  ( prev_best   )

    return new_lenghts, new_times


def get_mean_times( df, et, est, comb ):

    f = comb.split(' ')

    if f[0] == 'METAL':
        df_row_filtered = df[ (df['data element type'] == et ) & (df['data element subtype'] == est ) & (df['implementation type'] == f[0] ) & (df['metal implementation type'] == f[1] ) ]
    else:
        df_row_filtered = df[ (df['data element type'] == et ) & (df['data element subtype'] == est ) & (df['implementation type'] == f[0] ) & (df['loop unrolling factor'] == int(f[1]) ) & (df['num CPU threads'] == int(f[2]) ) ]

    if est == 'VECTOR' or est == 'STRUCTURE_OF_ARRAYS' or est == 'ARRAY_OF_STRUCTURES':

        vector_lengths = [ int(r['vector length/matrix row']) for i, r in df_row_filtered.iterrows() ]

    elif est == 'MATRIX_COL_MAJOR' or est == 'MATRIX_ROW_MAJOR':

        vector_rows = [ int(r['vector length/matrix row']) for i, r in df_row_filtered.iterrows() ]
        vector_cols = [ int(r['matrix columns']) for i, r in df_row_filtered.iterrows() ]
        vector_lengths = []
        for i, r in enumerate(vector_rows):
            c = vector_cols[i]
            vector_lengths.append(r*c)

    elif est == 'MATRIX_SPARSE':
        vector_lengths = [ int(r['number of non zeros']) for i, r in df_row_filtered.iterrows() ]

    mean_times     = [ float(r['mean time milliseconds'])   for i, r in df_row_filtered.iterrows() ]
    
    return get_best_times_for_impl( vector_lengths, mean_times )

def get_x_label(est):
    if est == 'VECTOR' or est == 'STRUCTURE_OF_ARRAYS' or est == 'ARRAY_OF_STRUCTURES':

        return '|Vec|'

    elif est == 'MATRIX_COL_MAJOR' or est == 'MATRIX_ROW_MAJOR':

        return '|Mat|'

    elif est == 'MATRIX_SPARSE':

        return 'NNZ'

def plot_log_log( element_type, element_subtype, width, height, title, lengths, times, labels ):

    fig = plt.figure( figsize = (width, height) )
    ax1 = fig.add_subplot(111)

    ax1.set_yscale('log')
    ax1.set_xscale('log', base=10)
    ax1.set_title(title)
    plt.xlabel( get_x_label(element_subtype) )
    plt.ylabel("[ms]")
    for i, length in enumerate(lengths):
        my_time = times[i]
        label  = labels[i]
        ax1.plot( length, my_time, label=label, marker='o' )

    ax_handles, ax_labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(ax_handles, ax_labels)
    ax1.grid('on')

    plt.savefig(BASE_DIR + element_type + '_' + element_subtype + '_' + title.replace(' ','_').replace('+', 'P') + '.png')
    plt.clf()


def plot_log_lin_relative( element_type, element_subtype, width, height, title, lengths, times, labels, base_label, upper_limit ):

    fig = plt.figure( figsize = (width, height) )
    ax1 = fig.add_subplot(111)

    ax1.set_yscale('linear')
    ax1.set_ylim(0, upper_limit)
    ax1.set_xscale('log', base=10)
    ax1.set_title(title)
    plt.xlabel( get_x_label(element_subtype) )
    plt.ylabel("ratio")
    for i, length in enumerate(lengths):
        my_time = times[i]
        label  = labels[i]
        if label == base_label:
            label = label + "*"
        ax1.plot( length, my_time, label=label, marker='o' )
    ax_handles, ax_labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(ax_handles, ax_labels)
    ax1.grid('on')

    plt.savefig(BASE_DIR + element_type + '_' + element_subtype + '_' + title.replace(' ', '_') + '_relative.png')
    plt.clf()



def plot_lin_lin( element_type, element_subtype, width, height, title, lengths, times, labels ):

    fig = plt.figure( figsize = (width, height) )
    ax1 = fig.add_subplot(111)

    ax1.set_yscale('linear')
    ax1.set_xscale('linear')
    ax1.set_title(title)
    plt.xlabel( get_x_label(element_subtype) )
    plt.ylabel("[ms]")
    for i, length in enumerate(lengths):
        my_time = times[i]
        label  = labels[i]
        ax1.plot( length, my_time, label=label, marker='o' )

    ax_handles, ax_labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(ax_handles, ax_labels)
    ax1.grid('on')

    plt.savefig(BASE_DIR + element_type + '_' + element_subtype + '_' + title.replace(' ','_').replace('+', 'P') + '.png')
    plt.clf()


def show_implementations(df):

    data_element_types     = get_unique_values_for_column( df, 'data element type'    )
    data_element_subtypes  = get_unique_values_for_column( df, 'data element subtype' )

    for et in data_element_types:

        for est in data_element_subtypes:

            print ('')
            print ( 'Element Type: [' + et + '] Subtype: [' + est + ']' )
            df_per_data_type = df[ (df['data element type'] == et ) & (df['data element subtype'] == est ) ]

            vector_lengths = get_unique_values_for_column( df_per_data_type, 'vector length/matrix row' )
            print ('Vector Lengths: ' + str(vector_lengths))

            cols = get_unique_values_for_column( df_per_data_type, 'matrix columns' )
            print ('Cols: ' + str(cols))

            nnz = get_unique_values_for_column( df_per_data_type, 'number of non zeros' )
            print ('Number of Nonzeros: ' + str(nnz))

            implementation_types   = get_unique_values_for_column( df_per_data_type, 'implementation type'      )
            num_cpu_threads        = get_unique_values_for_column( df_per_data_type, 'num CPU threads'          )
            loop_unrolling_factors = get_unique_values_for_column( df_per_data_type, 'loop unrolling factor'    )

            implementation_combinations = get_implementation_combinations( df_per_data_type )
            out_str = ""
            for i, ic in enumerate(implementation_combinations):
                if i > 0:            
                    out_str += ", "
                out_str += "\""
                out_str += ic
                out_str += "\""
            print ('Implementation Combinarions: ' + out_str )


def plot_charts( df, js ):

    for plot_data in js:
        
        width  = int( plot_data[ 'Width'  ] )
        height = int( plot_data[ 'Height' ] )
        title  = plot_data[ 'Title' ]

        list_lengths = []
        list_times   = []
        list_labels  = []

        det = plot_data[ 'DataElementType'    ]
        des = plot_data[ 'DataElementSubtype' ]
        pt  = plot_data[ 'PlotType'           ]
        ul  = float(plot_data[ 'UpperLimit' ])
        base_mapping = {}

        if des != 'ANY':

            if pt == 'LOG-LIN-RELATIVE':
                pb = plot_data[ 'PlotBase' ]
                base_lengths, base_times = get_mean_times( df, det, des, pb )
                for i in range(len(base_lengths)):
                    base_mapping[ base_lengths[i] ] = base_times[i]
            pcs = plot_data[ 'PlotCases' ]
            for pc in pcs:

                lengths, times = get_mean_times( df, det, des, pc )

                if  pt == 'LOG-LIN-RELATIVE':

                    if len( base_times ) < len( times ):

                        print ('Number of elements does not match to the base.')
                        exit(1)

                    relative_times = []

                    for i, t in enumerate( times ):
                        L = lengths[i]
                        b = base_mapping[L]
                        relative_times.append( t / b )

                    list_times.append( relative_times )

                else:
                    list_times.append( times )

                list_lengths.append( lengths )
                list_labels.append( pc )
        else:

            if pt == 'LOG-LIN-RELATIVE':
                des, pb = plot_data[ 'PlotBase' ].split(':')
                base_lengths, base_times = get_mean_times( df, det, des, pb )
                for i in range(len(base_lengths)):
                    base_mapping[ base_lengths[i] ] = base_times[i]

            pcs_des = plot_data[ 'PlotCases' ]
            for pd in pcs_des:

                des2, pc = pd.split(':')

                lengths, times = get_mean_times( df, det, des2, pc )

                if  pt == 'LOG-LIN-RELATIVE':

                    if len( base_times ) < len( times ):

                        print ('Number of elements does not match to the base.')
                        exit(1)

                    relative_times = []

                    for i, t in enumerate( times ):
                        L = lengths[i]
                        b = base_mapping[L]
                        relative_times.append( t / b )

                    list_times.append( relative_times )

                else:
                    list_times.append( times )

                list_lengths.append( lengths )
                list_labels.append( pd )

        if pt == 'LOG-LIN-RELATIVE':
            plot_log_lin_relative( det, des, width, height, title, list_lengths, list_times, list_labels, pb, ul )
        elif  pt == 'LOG-LOG':
            plot_log_log( det, des, width, height, title, list_lengths, list_times, list_labels )
        elif  pt == 'LIN-LIN':
            plot_lin_lin( det, des, width, height, title, list_lengths, list_times, list_labels )

def main():

    comm_parser = parse_commandline()
    comm_args   = comm_parser.parse_args()

    df = pd.read_csv( comm_args.logfile, sep = "\t" )

    if comm_args.show_impl:

        show_implementations(df)

    if comm_args.plot_charts:
        with open(comm_args.specfile, 'r') as jf:
            js = json.load(jf)
            plot_charts(df, js)

        
if __name__ == "__main__":
    main()
