#!/usr/bin/env python

import os, io, random
import string
import numpy as np

from Bio.Seq import Seq
from Bio.Align import MultipleSeqAlignment
from Bio import AlignIO, SeqIO
from io import StringIO

import panel as pn
import panel.widgets as pnw

import pandas as pd
pn.extension()

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Plot, Grid, Range1d
from bokeh.models.glyphs import Text, Rect
from bokeh.layouts import gridplot
# from bokeh.palettes import mpl
from bokeh.palettes import brewer

import random
from Bio import pairwise2
from bokeh.io import export_svgs, export_png
from tqdm import tqdm

from Bio.Align.Applications import MuscleCommandline

msa_size = 10
func_filename='functional_gen_sps_200129.csv'

def view_alignment(aln, fontsize="9pt", plot_width=800):
    """Bokeh sequence alignment view"""

    #make sequence and id lists from the aln object
    ids = [rec.id for rec in aln]
    seqs = [rec.seq for rec in aln]
    ids, seqs = zip(*sorted(zip(ids, seqs), reverse=True))

    text = [i for s in list(seqs) for i in s]
    colors = get_colors(seqs)
#     breakpoint()
    N = len(seqs[0])
    S = len(seqs)
    width = .5

    x = np.arange(1,N+1)
    y = np.arange(0,S,1)
    #creates a 2D grid of coords from the 1D arrays
    xx, yy = np.meshgrid(x, y)
    #flattens the arrays
    gx = xx.ravel()
    gy = yy.flatten()
    #use recty for rect coords with an offset
    recty = gy+.5
    h= 1/S
    #now we can create the ColumnDataSource with all the arrays
    source = ColumnDataSource(dict(x=gx, y=gy, recty=recty, text=text, colors=colors))
    plot_height = len(seqs)*15+50
    x_range = Range1d(0.5,N+1.5, bounds='auto')
    if N>100:
        viewlen=100
    else:
        viewlen=N
    #view_range is for the close up view
    view_range = (0.5,viewlen + 0.5)
    tools="xpan, xwheel_zoom, reset, save"

    #entire sequence view (no text, with zoom)
    p = figure(title=None, plot_width= plot_width, plot_height=50,
               x_range=x_range, y_range=(0,S), tools=tools,
               min_border=0, toolbar_location='below')
    rects = Rect(x="x", y="recty",  width=1, height=1, fill_color="colors",
                 line_color=None, fill_alpha=0.6)
    p.add_glyph(source, rects)
    p.yaxis.visible = False
    p.grid.visible = False

    #sequence text view with ability to scroll along x axis
    p1 = figure(title=None, plot_width=plot_width, plot_height=plot_height,
                x_range=view_range, y_range=ids, tools="xpan,reset, previewsave",
                min_border=0, toolbar_location='below', output_backend="svg")#, lod_factor=1)
    rects = Rect(x="x", y="recty",  width=1, height=1, fill_color="colors",
                line_color=None, fill_alpha=0.5)
    glyph = Text(x="x", y="y", text="text", text_align='center',text_color="black",
                text_font="monospace",text_font_size=fontsize)
    p1.add_glyph(source, glyph)
    p1.add_glyph(source, rects)

    p1.grid.visible = False
    p1.xaxis.major_label_text_font_style = "bold"
    p1.yaxis.minor_tick_line_width = 0
    p1.yaxis.major_tick_line_width = 0

    p = gridplot([[p],[p1]], toolbar_location='below')
    return gridplot([[p1]])

def get_colors(seqs):
    """make colors for bases in sequence"""
    text = [i for s in list(seqs) for i in s]

#     aas = list("ACDEFGHIKLMNPQRSTVWY-")

    # ClustalW-like splits
    # http://www.jalview.org/help/html/colourSchemes/clustal.html
    clrs = {}
    ref_colors = brewer['Paired'][8] # https://docs.bokeh.org/en/latest/docs/reference/palettes.html
    ref_colors.sort()
    random.seed(26)
    random.shuffle(ref_colors)
    for aa in list('AILMFWV'):
        clrs.update({aa:ref_colors[0]})
    for aa in list('KR'):
        clrs.update({aa:ref_colors[1]})
    for aa in list('DE'):
        clrs.update({aa:ref_colors[2]})
    for aa in list('NQST'):
        clrs.update({aa:ref_colors[3]})
    clrs.update({'C':ref_colors[4]})
    clrs.update({'G':ref_colors[5]})
    clrs.update({'P':ref_colors[6]})
    clrs.update({'H':ref_colors[7]})
    clrs.update({'Y':ref_colors[7]})
    clrs.update({'-':'white'})

    colors = [clrs[i] for i in text]
    return colors


df = pd.read_csv('sp_top_100_scores.csv')

df_func = pd.read_csv(func_filename)
func_sps = list(set(df_func['seq'].values))

closest_matches = []
closest_identities = []

for sample_sp in tqdm(func_sps):

    _df = df[df['generated'] == sample_sp]
    _df

    # Generate .fasta input
    gen_sp = sample_sp
    fasta_filename = 'fasta/' + gen_sp + '.fasta'
    aln_filename = 'alns/' + gen_sp + '.aln'
    with open(fasta_filename, 'w+') as f:
        f.write('>gen_sp\n')
        f.write(gen_sp + '\n')

        sps = _df[:msa_size]['natural'].values
        for i, sp in enumerate(sps):
            f.write(f'>nat_sp{i}\n')
            f.write(f'{sp}\n')

    # Make MUSCLE alignment
    # http://biopython.org/DIST/docs/tutorial/Tutorial.html#htoc81
    from Bio.Align.Applications import MuscleCommandline
    muscle_exe = r"/home/wuzachar/bin/muscle3.8.31_i86linux64"
    muscle_cline = MuscleCommandline(muscle_exe, input=fasta_filename, out=aln_filename)
    stdout, stderr = muscle_cline()
    # aln = list(AlignIO.parse(aln_filename, "fasta"))

    # Get percent identity
    gen_sp = _df.iloc[0]['generated']
    nat_sp = _df.iloc[0]['natural']

    actual_sp = gen_sp.replace('-','')
    actual_length = len(actual_sp)

    alignments = pairwise2.align.globalms(gen_sp, nat_sp, 2, -1, -1, -1)
    aln_gen, aln_nat, score, _, _ = alignments[0]

    match_count = 0
    for i in range(len(aln_gen)):
        if aln_gen[i] != '-':
            if aln_gen[i] == aln_nat[i]:
                match_count += 1

    print(aln_gen)
    print(aln_nat)
    percent_identity = match_count / actual_length
    print(match_count, actual_length, f"{percent_identity*100:0.2f} % identity")

    # Export alignments
    # aln = AlignIO.read('alns/sample_aln.fasta','fasta')
    aln = AlignIO.read(aln_filename,'fasta')
    p = view_alignment(aln, plot_width=800)
    export_svgs(p, filename="figs/" + sample_sp + ".svg")
#     export_png(p, filename="figs/" + sample_sp + ".png")
    # pn.pane.Bokeh(p)
    # export_svgs(p, filename="figs/" + sample_sp + ".svg")

    closest_matches.append(aln_nat.replace('-',''))
    closest_identities.append(percent_identity)

# write final file
df_func['closest_match'] = closest_matches
df_func['percent_identity'] = closest_identities
df_func.to_csv('alns/func_sps_matches.csv')
