---
jupyter:
  hide_input: false
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.8.8
  nbformat: 4
  nbformat_minor: 2
  toc:
    base_numbering: 1
    nav_menu: {}
    number_sections: true
    sideBar: true
    skip_h1_title: false
    title_cell: Table of Contents
    title_sidebar: Contents
    toc_cell: false
    toc_position:
      height: calc(100% - 180px)
      left: 10px
      top: 150px
      width: 196.274px
    toc_section_display: true
    toc_window_display: false
  varInspector:
    cols:
      lenName: 16
      lenType: 16
      lenVar: 40
    kernels_config:
      python:
        delete_cmd_prefix: del
        library: var_list.py
        varRefreshCmd: print(var_dic_list())
      r:
        delete_cmd_postfix: )
        delete_cmd_prefix: rm(
        library: var_list.r
        varRefreshCmd: cat(var_dic_list())
    oldHeight: 489.31922199999997
    position:
      height: 40px
      left: 702.8px
      right: 20px
      top: 51px
      width: 524.708px
    types_to_exclude:
    - module
    - function
    - builtin_function_or_method
    - instance
    - \_Feature
    varInspector_section_display: none
    window_display: true
---

::: {.cell .code execution_count="22" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
%matplotlib inline
```
:::

::: {.cell .code execution_count="23" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
import matplotlib.pyplot as plt 
import pandas as pd               # Python data science library
import numpy as np
import re
import sys

from modelclass import model
import modelpattern as pt
import modelmanipulation as mp    # Module for model text processing
from modelmanipulation import explode
```
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
# ModelFlow, A library to manage and solve Models

CEF Virtual Conference 202

Ib Hansen `<br>`{=html} <Ib.hansen.iv@gmail.com>

Work done at Danmarks Nationalbank and ECB.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
**Problem**:`<br>`{=html} Stress-test model for banks`<br>`{=html}
Complicated and slow **Excel** workbook `<br>`{=html} Difficult to
maintain and change
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"subslide\"}"}
**Solution**:`<br>`{=html} Separate specification (at a high level of
abstraction) and solution code.`<br>`{=html} Python comes **batteries
included**. Data management, text processing, visualization \...
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"fragment\"}"}
**Implementation** of **minimum workable toolkit** `<br>`{=html} Create
a **transpiler**: A model in a domain specific Business logic
language:`<br>`{=html}
`frml <> loss = probability_of_default * loss_given_default * exposure(-1)``<br>`{=html}
==\> python model code`<br>`{=html}
`values[row,0]=values[row,1]*values[row,2]*values[row-1,3]``<br>`{=html}`<br>`{=html}
Create **solver** and **utility** functions using Python libraries.
`<br>`{=html} Data wrangling: **Pandas**`<br>`{=html} A tokenizer:
**re** `<br>`{=html} Analyze logical structure:
**Networkx**`<br>`{=html} \...
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"fragment\"}"}
**Refactor and refine**`<br>`{=html} Larger models, Faster transpiler,
Newton and Gauss solvers, Logical structure, Derivatives, Visualization,
Front ends \...
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
## Refine and refactor

To suit the needs of the different models thrown at the toolkit.

**Specify very large (or small) models** as concise and intuitive
equations. The user don\'t have to do the household chores and can
concentrate on the economic content.

**Large models**. 1 million equation and more can be handled.

**Agile model development** Model are specified at a high level of
abstraction and are processed fast. Experiments with model specification
are agile and fast.

**Onboarding models and combining from different sources**. Recycling
and combining models specified in different ways: Excel, Latex, Dynare,
Python or other languages. Python\'s ecosystem makes it possible to
transform many different models into ModelFlow models or to wrap them
into functions which can be called from ModelFlow models.

**Onboarding data from different sources**. Pythons Pandas Library and
other tools are fast and efficient for data management.

**A rich set of analytical tools for model and result analytic** helps
to understand the model and its results.

**The user can extend and modify the tools** to her or his needs. All
code is in Python and the core is quite small.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"subslide\"}"}
![](image-2.png)
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
## What is a Model in ModelFlow

ModelFlow is created to handle models. The term
[**model**](https://en.wikipedia.org/wiki/Model) can mean a lot of
different concepts.

The scope of models handled by ModelFlow is **discrete** models which is
the same for each time frame, can be formulated as **mathematical
equations** and *can* have **lagged** and **leaded** variables.
`<br>`{=html} This allows the system to handle quite a large range of
models.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
A model with:

-   $\textbf n$ number of endogeneous variables
-   $\textbf k$ number of exogeneous variables
-   $\textbf u$ max lead of endogeneous variables
-   $\textbf r$ max lag of endogeneous variables
-   $\textbf s$ max lag of exogeneous variables
-   $t$ time frame (year, quarter, day second or another another unit)

can be written in two ways, **normalized** or **un-normalized** form
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"subslide\"}"}
### Normalized model

Each endogenous variable is on the left hand side one time - and only
one time.`<br>`{=html} \\begin{eqnarray} y_t\^1 & = &
f\^1(y\_{t+u}\^1\...,y\_{t+u}\^n\...,y_t\^2\...,y\_{t}\^n\...y\_{t-r}\^1\...,y\_{t-r}\^n,x_t\^1\...x\_{t}\^k,\...x\_{t-s}\^1\...,x\_{t-s}\^k)
\\ y_t\^2 & = &
f\^2(y\_{t+u}\^1\...,y\_{t+u}\^n\...,y_t\^1\...,y\_{t}\^n\...y\_{t-r}\^1\...,y\_{t-r}\^n,x_t\^1\...x\_{t}\^k,\...x\_{t-s}\^1\...,x\_{t-s}\^k)
\\ \\vdots \\ y_t\^n & = &
f\^n(y\_{t+u}\^1\...,y\_{t+u}\^n\...,y_t\^1\...,y\_{t}\^{n-1}\...y\_{t-r}\^1\...,y\_{t-r}\^n,x_t\^1\...x\_{t}\^r,x\...\_{t-s}\^1\...,x\_{t-s}\^k)
\\end{eqnarray}

Many stress test, liquidity, macro or other models conforms to this
pattern. Or the can easily be transformed to thes pattern.

Written in matrix notation where $\textbf{y}_t$ and $\textbf{x}_t$ are
vectors of endogenous/exogenous variables for time t`<br>`{=html}

\\begin{eqnarray}\
\\textbf{y}*t & = & \\textbf{F}(\\textbf{y}*{t+u} \\cdots \\textbf{y}*t
\\cdots \\textbf{y}*{t-r},\\textbf{x}*t \\cdots \\textbf{x}*{t-s})
\\end{eqnarray}

ModelFlow allows variable (the 𝐱 \'es and the 𝐲 \'es the to be scalars,
matrices, arrays or pandas dataframes.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"subslide\"}"}
### Un-normalized form

Some models can not easy be specified as normalized formulas. Especially
models with equilibrium conditions can more suitable be specified in the
more generalized un-normalized form.

Written in matrix notation like before:

\\begin{eqnarray}\
\\textbf{0}& = & \\textbf{F}(\\textbf{y}\_{t+u} \\cdots \\textbf{y}*t
\\cdots \\textbf{y}*{t-r},\\textbf{x}*t \\cdots \\textbf{x}*{t-s})
\\end{eqnarray}

The number of endogenous variables and equations should still be the
same.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
#### Model solution

For a normalized model:

\\begin{eqnarray}\
\\textbf{y}*t & = & \\textbf{F}(\\textbf{y}*{t+u} \\cdots \\textbf{y}*t
\\cdots \\textbf{y}*{t-r},\\textbf{x}*t \\cdots \\textbf{x}*{t-r})\
\\end{eqnarray}

a solution is $\textbf{y}_t^*$ so that:

\\begin{eqnarray}\
\\textbf{y}*t\^\* & = & \\textbf{F}(\\textbf{y}*{t+u} \\cdots
\\textbf{y}*t\^\* \\cdots \\textbf{y}*{t-r},\\textbf{x}*t \\cdots
\\textbf{x}*{t-r})\
\\end{eqnarray}
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"fragment\"}"}
For the un-normalized model: \\begin{eqnarray}\
\\textbf{0}& = & \\textbf{F}(\\textbf{y}\_{t+u} \\cdots \\textbf{y}*t
\\cdots \\textbf{y}*{t-r},\\textbf{x}*t \\cdots \\textbf{x}*{t-s})
\\end{eqnarray}

a solution $\textbf{y}_t^*$ is

\\begin{eqnarray}\
\\textbf{0} & = & \\textbf{G}(\\textbf{y}\_{t+u} \\cdots
\\textbf{y}*t\^\* \\cdots \\textbf{y}*{t-r},\\textbf{x}*t \\cdots
\\textbf{x}*{t-r})\
\\end{eqnarray}

Some models can have more than one solution. In this case the solution
can depend on the starting point of the solution algorithm.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
## Model derivatives

Both for solving and for analyzing the causal structure of a model it
can be useful to define different matrices of derivatives for a model
$\textbf F()$ like this:

\\begin{align}\
\\textbf{A}\_t & = & \\frac{\\partial \\textbf{F}}{\\partial
\\textbf{y}\_t\^T} \\hphantom{\\hspace{5 mm} i=1, \\cdots , r}\
&\\hspace{1 mm}\\mbox{Derivatives with respect to current endogeneous
variables} \\ \\ \\textbf{E}*t\^i & = & \\frac{\\partial
\\textbf{F}}{\\partial \\textbf{y}*{t-i}\^T } \\hspace{5 mm} i=1,
\\cdots , r &\\hspace{1 mm}\\mbox{ Derivatives with respect to lagged
endogeneous variables } \\ \\ \\textbf{D}*t\^j & = & \\frac{\\partial
\\textbf{F}}{\\partial \\textbf{y}*{t+j}\^T } \\hspace{5 mm} j=1,
\\cdots , u &\\hspace{1 mm}\\mbox{ Derivatives with respect to leaded
endogeneous variables } \\ \\ \\textbf{F}*t\^k & = & \\frac{\\partial
\\textbf{F}}{\\partial \\textbf{x}*{t-i} \^T} \\hspace{5 mm} k=0,
\\cdots , s &\\hspace{1 mm}\\mbox{ Derivatives with respect to current
and lagged exogeneous variables }\\ \\end{align}

For un-normalized models the derivative matrices are just the dervatives
of $\textbf G$ instead of $\textbf F$
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
![](image.png)
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
## Model solutions
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"notes\"}"}
There are numerous methods to solve models (systems) as mentioned above.
ModelFlow can apply 3 different types of model solution methods:

1.  If the model has **no contemporaneous feedback**, the equations can
    be sorted
    [Topological](https://en.wikipedia.org/wiki/Topological_sorting) and
    then the equations can be calculated in the topological order. This
    is the same as a spreadsheet would do.\
2.  If the model has **contemporaneous feedback** model is solved with
    an iterative method. Here variants of well known solution methods
    are used:
    1.  [Gauss-Seidle](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method)
        (**Gauss**) which can handle large systems, is fairly robust and
        don\'t need the calculation of derivatives
    2.  [Newthon-Raphson](https://en.wikipedia.org/wiki/Newton%27s_method)
        (**Newton**) which requires the calculation of derivatives and
        solving of a large linear system but typically converges in
        fewer iterations.

Nearly all of the models solved by ModelFlow don\'t contain leaded
endogenous variables. Therefor they can be solved one period at a time.
For large sparse nonlinear models Gauss works fine. It solves a model
quite fast and we don\'t need the additional handiwork of handling
derivatives and large linear systems that Newton methods require.
Moreover many models in question do not have smooth derivatives. The
order in which the equation are calculated can have a large impact on
the convergence speed.

For some models the Newton algorithm works better. Some models are not
able to converge with Gauss-Seidle other models are just faster using
Newton. Also the ordering of equations does not matter for the
convergence speed.

However some models like FRB/US and other with **rational expectations**
or **model consistent expectations** contains leaded endogenous
variables. Such models typical has to be solved as one system for for
all projection periods. In this case, the Gauss variation
[Fair-Taylor](https://fairmodel.econ.yale.edu/rayfair/pdf/1983A.PDF) or
Stacked-Newton Method. The **stacked Newton** methods can be used in all
cases, but if not needed, it will usually use more memory and be slower.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"fragment\"}"}
        Newton            Gauss
  ------------------ ----------------
   ![](image-2.png)   ![](image.png)
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"fragment\"}"}
  Model           No contemporaneous feedback   Contemporaneous feedback        Leaded variables
  --------------- ----------------------------- ------------------------------- ---------------------------------------------
  Normalized      Calculate                     Gauss or `<br>`{=html} Newton   Fair Taylor or `<br>`{=html} Stacked Newton
  Un-normalized   Newton                        Newton                          Stacked Newton
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
# Implementation of solving algorithms in Python

Solving a model entails a number of steps:

1.  Specification of the model
2.  Create a dependency graph.
3.  Establish a solve order and separate the the model into smaller
    sub-models
4.  Create a python function which can evaluating
    $f_i(y_1^{k},\cdots,y_{i-1}^{k},y_{i+1}^{k-1},\cdots,y_{n}^{k-1},z)$
5.  If needed, create a python function which can evaluate the
    Jacobimatrices: $\bf{A,E,D}$ or $\bf{\bar A,\bar E,\bar D}$
6.  Apply a solve function (Gauss or Newton) using the elements above to
    the data.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
### Normalized model {#normalized-model}
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
#### Calculation, No contemporaneous feedback

In systems with no lags each period can be solved in succession The
equations has to be evaluated in a logical (topological sorted) order.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
Let $z$ be all predetermined values: all exogenous variable and lagged
endogenous variable.

Order the $n$ endogeneous variables in topological order.

For each time period we can find a solution by

for $i$ = 1 to $n$

> $y_{i}^{k} = f_i(y_1^{k},\cdots,y_{i-1}^{k},y_{i+1}^{k-1},\cdots,y_{n}^{k-1},z)$
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
#### The Gauss-Seidel algorithm. Normalized models with contemporaneous feedback {#the-gauss-seidel-algorithm-normalized-models-with-contemporaneous-feedback}

The Gauss-Seidel algorithm is quite straight forward. It basically
iterate over the formulas, until convergence.

let:`<br>`{=html} $z$ be all predetermined values: all exogenous
variable and lagged endogenous variable.`<br>`{=html} $n$ be the number
of endogenous variables.`<br>`{=html} $\alpha$ dampening factor which
can be applyed to selected equations

For each time period we can find a solution by doing Gauss-Seidel
iterations:

for $k = 1$ to convergence

> for $i$ = 1 to $n$
>
> > $y_{i}^{k} = (1-\alpha) * y_{i}^{{k-1}} + \alpha f_i(y_1^{k},\cdots,y_{i-1}^{k},y_{i+1}^{k-1},\cdots,y_{n}^{k-1},z)$
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
#### The Newton-Raphson algorithme. Models with contemporaneous feedback {#the-newton-raphson-algorithme-models-with-contemporaneous-feedback}

Let $\bf z$ be a vector all predetermined values: all exogenous variable
and lagged endogenous variable.

For each time period we can find a solution by doing Newton-Raphson
iterations:`<br>`{=html}

for $k = 1$ to convergence`<br>`{=html}

> $\bf y = \bf {F(y^{k-1},z) }$
>
> $\bf y^{k} = \bf y - \alpha \times \bf{(A-I)}^{-1} \times ( \bf {y - y^{k-1} })$

The expression: $\bf{(A-I)}^{-1}\times ( \bf {y - y^{k-1} })$ is the
same as the solution to:

$\bf {y- y^{k-1} } = \bf (A-I) \times \bf x$

This problem can be solved much more efficient than performing
$\bf{(A-I)}^{-1}\times ( \bf {y - y^{k-1} })$

The Scipy library provides a number of solvers to this linear set of
equations. There are both solvers using factorization and iterative
methods, and there are solvers for dense and sparce matrices. All linear
solvers can easily be incorporated into ModelFlows Newton-Raphson
nonlinear solver.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
#### Stacked Newton-Raphson all periods in one go. Models with both leaded and lagged endogeneous variable {#stacked-newton-raphson-all-periods-in-one-go-models-with-both-leaded-and-lagged-endogeneous-variable}

If the model has leaded endogenous variables it can in general not be
solved one time period at a time. We have to solve the model for all
time frames as one large model.

$$\bf{\bar A} =\begin{bmatrix} 
		\bf{A_1}   & \bf{D_1^1} & \bf{D_1^2} & \bf{0}     &\bf{0}      &\bf{0}      &\bf{0}      &\bf{0}  \\
        \bf{E_2^1} & \bf{A_2}   & \bf{D_2^1} & \bf{D_2^2} &\bf{0}      &\bf{0}      &\bf{0}      &\bf{0} \\       
        \bf{E_3^2} & \bf{E_3^1} & \bf{A_3}   & \bf{D_3^1} & \bf{D_3^2} &\bf{0}      &\bf{0}      &\bf{0} \\       
        \bf{E_4^3} & \bf{E_4^2} & \bf{E_4^1} & \bf{A_4}   & \bf{D_4^1} & \bf{D_4^2} &\bf{0}      & \bf{0} \\       
        \bf{0}     & \bf{E_5^3} & \bf{E_5^2} & \bf{E_5^1} & \bf{A_5}   & \bf{D_5^1} & \bf{D_5^2} &\bf{0}\\       
        \bf{0}     & \bf{0}     & \bf{E_6^3} & \bf{E_6^2} & \bf{E_6^1} & \bf{A_6}   & \bf{D_6^1} & \bf{D_6^2}\\       
        \bf{0}     & \bf{0}     & \bf{0}     & \bf{E_7^3} & \bf{E_7^2} & \bf{E_7^1} & \bf{A_7}   & \bf{D_7^1} \\       
        \bf{0}     & \bf{0}     & \bf{0}     & \bf{0}     & \bf{E_8^3} & \bf{E_8^2} & \bf{E_8^1} & \bf{A_8} \\       
\end{bmatrix} \bar y = \begin{bmatrix}\bf{y_1}\\\bf{y_2}\\\bf{y_3}\\ \bf{y_4} \\\bf{y_5} \\\bf{y_6} \\ \bf{y_7} \\ \bf{y_8} \end{bmatrix} \bar F = \begin{bmatrix}\bf{F}\\\bf{F}\\\bf{F}\\ \bf{F} \\\bf{F} \\\bf{F} \\ \bf{F} \\ \bf{F} \end{bmatrix}$$
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
Now the solution algorithme looks like this.

Again let $\bf z$ be a vector all predetermined values: all exogenous
variable and lagged endogenous variable.

> for $k = 1$ to convergence`<br>`{=html}
>
> > $\bf{\bar y} = \bf {\bar F(\bar y^{k-1},\bar z) }$`<br>`{=html}
> > $\bf {\bar y^{k}} = \bf{\bar y} - \alpha \times \bf{(\bar A-I)}^{-1}\times ( \bf {\bar y - \bar y^{k-1} })$

Notice that the model $\bf F$ is the same for all time
periods.`<br>`{=html} However, as time can be an exogenous variable the
result of $\bf{F}$ can depend on time. This allows us to specify
terminal conditions.

The update frequency of $\bf{\bar A}$ and $\alpha$ and the value of
$\alpha$ can be set to manage the speed and stability of the algorithm.

We solve the problem:
$$( \bf {\bar y - \bar y^{k-1} }) = \bf{(\bar A-I)}\times \bf x $$
instead of inverting $\bf{A}$.

Python gives access to very efficient sparse libraries. The [Scipy
library](https://scipy.org/scipylib/index.html) utilizes the [Intel®
Math Kernel Library](https://software.intel.com/en-us/mkl). Any of the
available routines for solving linear systems can easily be
incorporated.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
## Create a model instance which calculates the Jacobi matrices. {#create-a-model-instance-which-calculates-the-jacobi-matrices}

The derivatives of all formulas with respect to all endogenous variables
are needed.

First step is to specifying a model in the business logic language which
calculate all the non-zero elements`<br>`{=html} In ModelFlow this can
be done by **symbolic**, by **numerical differentiation** or by a
combination.

The formula for calculating
$\dfrac{\partial{numerator }}{{\partial denominator(-lag)}}$ is written
as:

\< numerator \>\_\_p\_\_\< denominator \>\_\_lag\_\_\< lag\> =
derivative expression

Just another instance of a ModelFlow model class.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
### A small Solow model to show the construction of the Jacobi matrix. {#a-small-solow-model-to-show-the-construction-of-the-jacobi-matrix}

An example can be helpful`<br>`{=html} First a small model is defined -
in this case a solow growth model:
:::

::: {.cell .code execution_count="24" slideshow="{\"slide_type\":\"subslide\"}"}
``` {.python}
fsolow = '''\
Y         = a * k**alfa * l **(1-alfa) 
C         = (1-SAVING_RATIO)  * Y 
I         = Y - C 
diff(K)   = I-depreciates_rate * K(-1)
diff(l)   = labor_growth * L(-1) 
K_i= K/L '''
msolow = model.from_eq(fsolow)
```
:::

::: {.cell .code execution_count="25" slideshow="{\"slide_type\":\"fragment\"}"}
``` {.python}
print(msolow.equations)
```

::: {.output .stream .stdout}
    FRML <> Y         = A * K**ALFA * L **(1-ALFA)  $
    FRML <> C         = (1-SAVING_RATIO)  * Y  $
    FRML <> I         = Y - C  $
    FRML <> K=K(-1)+(I-DEPRECIATES_RATE * K(-1))$
    FRML <> L=L(-1)+(LABOR_GROWTH * L(-1))$
    FRML <> K_I= K/L  $
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
### Create some data and solve the model
:::

::: {.cell .code execution_count="26" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
N = 100
df = pd.DataFrame({'L':[100]*N,'K':[100]*N})
df.loc[:,'ALFA'] = 0.5
df.loc[:,'A'] = 1.
df.loc[:,'DEPRECIATES_RATE'] = 0.05
df.loc[:,'LABOR_GROWTH'] = 0.01
df.loc[:,'SAVING_RATIO'] = 0.05
msolow(df,max_iterations=100,first_test=10,silent=1);
```
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
### Create an differentiation instance of the model

Use symbolic differentiation when possible else use numerical
differentiation.
:::

::: {.cell .code execution_count="27" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
from modelnewton import newton_diff
msolow.smpl(3,5);  # we only want a few years 
```
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
### Symbolic differentiation
:::

::: {.cell .code execution_count="28" slideshow="{\"slide_type\":\"-\"}"}
``` {.python}
newton = newton_diff(msolow)
print(newton.diff_model.equations) 
```

::: {.output .stream .stdout}
    FRML  <> C__p__Y___lag___0 = 1-SAVING_RATIO   $
    FRML  <> I__p__C___lag___0 = -1   $
    FRML  <> I__p__Y___lag___0 = 1   $
    FRML  <> K__p__I___lag___0 = 1   $
    FRML  <> K__p__K___lag___1 = 1-DEPRECIATES_RATE   $
    FRML  <> K_I__p__K___lag___0 = 1/L   $
    FRML  <> K_I__p__L___lag___0 = -K/L**2   $
    FRML  <> L__p__L___lag___1 = LABOR_GROWTH+1   $
    FRML  <> Y__p__K___lag___0 = A*ALFA*K**ALFA*L**(1-ALFA)/K   $
    FRML  <> Y__p__L___lag___0 = A*K**ALFA*L**(1-ALFA)*(1-ALFA)/L   $
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
### Numerical differentiation
:::

::: {.cell .code execution_count="29" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
newton2 = newton_diff(msolow,forcenum=1)
print(newton2.diff_model.equations)
```

::: {.output .stream .stdout}
    FRML  <> C__p__Y___lag___0 = (((1-SAVING_RATIO)*(Y+0.0025))-((1-SAVING_RATIO)*(Y-0.0025)))/0.005   $
    FRML  <> I__p__C___lag___0 = ((Y-(C+0.0025))-(Y-(C-0.0025)))/0.005   $
    FRML  <> I__p__Y___lag___0 = (((Y+0.0025)-C)-((Y-0.0025)-C))/0.005   $
    FRML  <> K__p__I___lag___0 = ((K(-1)+((I+0.0025)-DEPRECIATES_RATE*K(-1)))-(K(-1)+((I-0.0025)-DEPRECIATES_RATE*K(-1))))/0.005   $
    FRML  <> K__p__K___lag___1 = (((K(-1)+0.0025)+(I-DEPRECIATES_RATE*(K(-1)+0.0025)))-((K(-1)-0.0025)+(I-DEPRECIATES_RATE*(K(-1)-0.0025))))/0.005   $
    FRML  <> K_I__p__K___lag___0 = (((K+0.0025)/L)-((K-0.0025)/L))/0.005   $
    FRML  <> K_I__p__L___lag___0 = ((K/(L+0.0025))-(K/(L-0.0025)))/0.005   $
    FRML  <> L__p__L___lag___1 = (((L(-1)+0.0025)+(LABOR_GROWTH*(L(-1)+0.0025)))-((L(-1)-0.0025)+(LABOR_GROWTH*(L(-1)-0.0025))))/0.005   $
    FRML  <> Y__p__K___lag___0 = ((A*(K+0.0025)**ALFA*L**(1-ALFA))-(A*(K-0.0025)**ALFA*L**(1-ALFA)))/0.005   $
    FRML  <> Y__p__L___lag___0 = ((A*K**ALFA*(L+0.0025)**(1-ALFA))-(A*K**ALFA*(L-0.0025)**(1-ALFA)))/0.005   $
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
### Display the full stacked matrix

To make the sparcity clear all zero values are shown as blank
:::

::: {.cell .code execution_count="30" scrolled="false" slideshow="{\"slide_type\":\"subslide\"}"}
``` {.python}
stacked_df = newton.get_diff_df_tot()
stacked_df.applymap(lambda x:f'{x:,.2f}' if x != 0.0 else ' ') 
```

::: {.output .execute_result execution_count="30"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>per</th>
      <th colspan="6" halign="left">3</th>
      <th colspan="6" halign="left">4</th>
      <th colspan="6" halign="left">5</th>
    </tr>
    <tr>
      <th></th>
      <th>var</th>
      <th>C</th>
      <th>I</th>
      <th>K</th>
      <th>K_I</th>
      <th>L</th>
      <th>Y</th>
      <th>C</th>
      <th>I</th>
      <th>K</th>
      <th>K_I</th>
      <th>L</th>
      <th>Y</th>
      <th>C</th>
      <th>I</th>
      <th>K</th>
      <th>K_I</th>
      <th>L</th>
      <th>Y</th>
    </tr>
    <tr>
      <th>per</th>
      <th>var</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">3</th>
      <th>C</th>
      <td>-1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.95</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>I</th>
      <td>-1.00</td>
      <td>-1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td>1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>K</th>
      <td></td>
      <td>1.00</td>
      <td>-1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>K_I</th>
      <td></td>
      <td></td>
      <td>0.01</td>
      <td>-1.00</td>
      <td>-0.01</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>L</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>-1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>Y</th>
      <td></td>
      <td></td>
      <td>0.51</td>
      <td></td>
      <td>0.49</td>
      <td>-1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">4</th>
      <th>C</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>-1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.95</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>I</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>-1.00</td>
      <td>-1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td>1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>K</th>
      <td></td>
      <td></td>
      <td>0.95</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1.00</td>
      <td>-1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>K_I</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.01</td>
      <td>-1.00</td>
      <td>-0.01</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>L</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1.01</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>-1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>Y</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.51</td>
      <td></td>
      <td>0.49</td>
      <td>-1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">5</th>
      <th>C</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>-1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>I</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>-1.00</td>
      <td>-1.00</td>
      <td></td>
      <td></td>
      <td></td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>K</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.95</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1.00</td>
      <td>-1.00</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>K_I</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.01</td>
      <td>-1.00</td>
      <td>-0.01</td>
      <td></td>
    </tr>
    <tr>
      <th>L</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1.01</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>-1.00</td>
      <td></td>
    </tr>
    <tr>
      <th>Y</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.51</td>
      <td></td>
      <td>0.49</td>
      <td>-1.00</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
### The results can also be displayed
:::

::: {.cell .code execution_count="31" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
newton.show_diff_latex()
```

::: {.output .display_data}
```{=latex}
$\begin{eqnarray*} Y_t  & = &  A_t \times K_t ** ALFA_t \times L_t ** ( 1 - ALFA_t ) \end{eqnarray*}$
```
:::

::: {.output .display_data}
```{=latex}
\begin{eqnarray*}\frac{\partial Y_t}{\partial K_t} & = & A_t \times ALFA_t \times K_t ** ALFA_t \times L_t ** ( 1 - ALFA_t ) / K_t\\\frac{\partial Y_t}{\partial L_t} & = & A_t \times K_t ** ALFA_t \times L_t ** ( 1 - ALFA_t ) \times ( 1 - ALFA_t ) / L_t\end{eqnarray*} 
```
:::

::: {.output .display_data}
    <IPython.core.display.Markdown object>
:::

::: {.output .display_data}
```{=latex}
$\begin{eqnarray*} L_t  & = &  L_{t-1} + ( LABOR\_GROWTH_t \times L_{t-1} ) \end{eqnarray*}$
```
:::

::: {.output .display_data}
```{=latex}
\begin{eqnarray*}\frac{\partial L_t}{\partial L_{t-1}} & = & LABOR\_GROWTH_t + 1\end{eqnarray*} 
```
:::

::: {.output .display_data}
    <IPython.core.display.Markdown object>
:::

::: {.output .display_data}
```{=latex}
$\begin{eqnarray*} K_t  & = &  K_{t-1} + ( I_t - DEPRECIATES\_RATE_t \times K_{t-1} ) \end{eqnarray*}$
```
:::

::: {.output .display_data}
```{=latex}
\begin{eqnarray*}\frac{\partial K_t}{\partial I_t} & = & 1\\\frac{\partial K_t}{\partial K_{t-1}} & = & 1 - DEPRECIATES\_RATE_t\end{eqnarray*} 
```
:::

::: {.output .display_data}
    <IPython.core.display.Markdown object>
:::

::: {.output .display_data}
```{=latex}
$\begin{eqnarray*} C_t  & = &  ( 1 - SAVING\_RATIO_t ) \times Y_t \end{eqnarray*}$
```
:::

::: {.output .display_data}
```{=latex}
\begin{eqnarray*}\frac{\partial C_t}{\partial Y_t} & = & 1 - SAVING\_RATIO_t\end{eqnarray*} 
```
:::

::: {.output .display_data}
    <IPython.core.display.Markdown object>
:::

::: {.output .display_data}
```{=latex}
$\begin{eqnarray*} I_t  & = &  Y_t - C_t \end{eqnarray*}$
```
:::

::: {.output .display_data}
```{=latex}
\begin{eqnarray*}\frac{\partial I_t}{\partial C_t} & = & - 1\\\frac{\partial I_t}{\partial Y_t} & = & 1\end{eqnarray*} 
```
:::

::: {.output .display_data}
    <IPython.core.display.Markdown object>
:::

::: {.output .display_data}
```{=latex}
$\begin{eqnarray*} K\_I_t  & = &  K_t / L_t \end{eqnarray*}$
```
:::

::: {.output .display_data}
```{=latex}
\begin{eqnarray*}\frac{\partial K\_I_t}{\partial K_t} & = & 1 / L_t\\\frac{\partial K\_I_t}{\partial L_t} & = & - K_t / L_t ** 2\end{eqnarray*} 
```
:::

::: {.output .display_data}
    <IPython.core.display.Markdown object>
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
## Speeding up solving through Just In Time compilation (Numba)

Python is an interpreted language. So slow

**Numba** is a Just In time Compiler.`<br>`{=html} Experience with a
Danish model (1700 equations) shows a speedup from 5 million floating
point operations per second (MFlops) to 800 MFlops. But compilation
takes time.

Also experiments with the **Cython** library has been performed. This
library will translate the Python code to C++ code. Then a C++ compiler
can compile the code and the run time will be improved a lot.

Also matrices can be used. This will force the use of the highly
optimized routines in the Numpy library.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
## Specification of model in Business Logic Language

The Business logic Language is a Python like language, where each
function $f_i$ from above is specified as:

    FRML <options> <left hand side> = <right hand side> $ ... 

The `<left hand side>` should not contain transformations, but can be a
tuple which match the `<right hand side>`. A \$ separates each formular.

Time is implicit, so $var_t$ is written as `var`, while $var_{t-1}$ is
written as `var(-1)` and $var_{t+1}$ is written as `var(+1)`. Case does
not matter. everything is eventual made into upper case.

It is important to be able to create short and expressive models,
therefor. Stress test models should be able to handle many bank and
sectors without repeating text. So on top of the **Business logic
language**. there is a **Macro Business Logic language**. The primary
goal of this is to allow (conditional) looping and normalization of
formulas.

The user can specify any conforming python function on the right hand
side
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
![](image.png)
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
# Onboarding a model

\*\*Python has incredible strong tools both for interacting with other
systems like Excel and Matlab. Some of the sources, from which models
has been recycled are:

-   Latex
    -   Model written in Latex - with some rules to allow text
        processing.
-   Eviews
-   Excel
    -   Calculation model from Excel workbook\
    -   Grabbing coefficients from excel workbooks
-   Matlab
    -   Wrapping matlab models into python functions, which can be used
        in ModelFlow\
    -   Grabbing coefficients from matlab .mat files.
-   Aremos models
-   TSP models

Grabbing models and transforming them to Business logic language usually
requires a tailor-made Python program. However in the ModelFlow folder
there are different examples of such grabbing.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
## The model structure

The logical structure of a model is useful for several reasons.

The structure of **contemporaneous endogenous variable** is used to
establish the calculation sequence and identify simultaneous systems
(strong graphcomponents).

The structure of a model can be seen as a directed graph. All variables
are node in the graph. If a variable $b$ is on the right side of the
formula defining variable $a$ there is an edge from $b$ to $a$.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
### First we define the nodes (vertices) of the dependency graph. {#first-we-define-the-nodes-vertices-of-the-dependency-graph}

The set of nodes is the set of relevant variables. Actually we want to
look at **two dependency graphs**: one containing *all variables*, and
one only containing *endogenous contemporaneous variable* (the
$y^j_t$\'s). So we define two sets S and E:

**All endogenous, exogenous, contemporaneous and lagged variables**

$S=\{y^j_{t-i}|j=1..n,i=1..r \} \cup \{x^j_{t-i}|j=1..k,i=1..s \}$
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
**Contemporaneous endogenous variables**

$E=\{y^j_{t}|j=1..n \}$

Naturally: $E \subseteq S$
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
### Then we define the edges of the dependency graph. {#then-we-define-the-edges-of-the-dependency-graph}

Again two sets are relevant:

**From all variables to contemporaneous endogenous variables**

$T = \{(a,b) | a \in E, b \in S\}$ a is on the right side of b
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
### And we can construct useful dependency graphs

The we can define a graph TG which defines the data dependency of the
model:

$TG = (S,T)$ The graph defined by nodes S and edges T.

TG can be used when exploring the dependencies in the model. This is
useful for the user when drilling down the results.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
However for preparing the solution a smaller graph has to be used. When
solving the model for a specific period both exogenous and lagged
endogenous variables are predetermined. Therefor we define the the
dependency graph for contemporaneous endogenous variables:

$TE = (E,T_e)$ The graph defined by nodes $S$ and edges $T_e$.

TE is used to determine if the model is simultaneous or not.

If the model is not simultaneous, then TE have no cycles, that is, it is
a Directed Acyclical Graph (DAG). Then we can find an order in which the
formulas can be calculated. This is called a topological order.

The topological order is a linear ordering of nodes (vertices) such that
for every edge (v,u), node v comes before u in the ordering.

A topological order is created by doing a topological sort of TE.

If TE, the dependency graph associated with F is **not** a Directed
Acyclical Graph (A DAG). Then F has contemporaneous feedback and is
simultaneous. Or - in Excel speak - the model has circular references.
And we need to use an iterative methods to solve the model. Sometime a
model contains several simultaneous blocks. Then each block is a strong
element of the graph. Each formula which is not part of a simultaneous
bloc is in itself a strong element.

A condensed graph where each strong element is condensed to a node is a
DAG. So the condensed graph have a topological order. This can be used
when solving the model.

The dependency graphs are constructed, analyzed and manipulated through
the **Networkx** Python library.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
#### Total dependency graphs

This shows $TG$ mentioned above.
:::

::: {.cell .code execution_count="32" slideshow="{\"slide_type\":\"slide\"}"}
``` {.python}
msolow.drawmodel(title='Total graph',all=0)
```

::: {.output .display_data}
![](16e9ecdb0fca05613db45113a12a062d9d469932.svg)
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
### The dependency graph for contemporaneous endogenous variables (TE)
:::

::: {.cell .code execution_count="33"}
``` {.python}
msolow.drawendo(title='Contemporaneous endo')
```

::: {.output .display_data}
![](4f2755125df61a146480e5454adf727f669e69c7.svg)
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
#### And the adjacency matrix of the graph

The graph can also be represented as a adjacency matrix. This is a is a
square matrix A. $A_{i,j}$ is one when there is an edge from node i to
node j, and zero when there is no edge.

If the graph is a DAG the adjacency matrix, and the elements are in a
topological order, is a lover triangular matrix.
:::

::: {.cell .code execution_count="34" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
a = msolow.plotadjacency(size=(8,8))
```

::: {.output .display_data}
![](c952a71ac2e7d3d4396eaa52c777b93f181b1acd.png)
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
## Solution ordering

### For normalized models:

For a model **without contemporaneous feedback**, the topoligical sorted
order is then used as calculating order.

For a model **with contemporaneous feedback** and no leaded variables,
ModelFlow divides a model into three parts. A recursive **prolog**
model, a recursive **epilog** model, the rest is the simultaneous
**core** model. Inside the core model the ordering of the equations are
preserved. It may be that the core model contains several strong
componens, which each could be solved as a simultanous system, however
it is solved as one simultanous system.

Only the core model is solved as a simultaneous system. The prolog model
is calculated once before the solving og the simultaneous system, the
epilog model is calculated once after the solution of the simultanous
system. For most models this significantly reduce the computational
burden of solving the model.

For a model with leaded variables where the model is stacked. All
equations are created equal.
:::

::: {.cell .code execution_count="35" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
# The preorder
print(f'The prolog variables {msolow.preorder}')
print(f'The core   variables {msolow.coreorder}')
print(f'The epilog variables {msolow.epiorder}')
```

::: {.output .stream .stdout}
    The prolog variables ['L']
    The core   variables ['Y', 'C', 'I', 'K']
    The epilog variables ['K_I']
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
# Some Model manipulation capabilities
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
## Model inversion aka Target/instruments or Goal Seek

In ordet to answer questions like:

-   How much capital has to be injected in order to maintain a certain
    GDP level in a stressed scenario?
-   How much loans has to be shredded by the banks in order to maintain
    a minimum level of capital (slim to fit)?
-   How much capital has to be injected in order to keep all bank above
    a certain capital threshold ?
-   What probability of transmission result in infected 2 weeks later
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"fragment\"}"}
The model instance is capable to **\"invert\"** a model. To use the
terminology of Tinbergen(1955) that is to calculate the value of some
exogenous variables - **the instruments** which is required in order to
achieve a certain target value for some endogenous variables - **the
targets**.

To use the terminology of Excel it is a goal/seek functionality with
multiple cells as goals and multiple cells as targets.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
The problem can be thought as follows: From the generic description of a
model: $\textbf{y}_t= \textbf{F}(\textbf{x}_{t})$. Here $\textbf{x}_{t}$
are all predetermined variables - lagged endogenous and all exogenous
variables.

It can be useful to allow a **delay**, when finding the instruments. In
this case we want to look at
$\textbf{y}_t= \textbf{F}(\textbf{x}_{t-delay})$

Think of a condensed model ($\textbf{G}$) with a few endogenous
variables($\bar{\textbf{y}}_t$): the targets and a few exogenous
variables($\bar{\textbf{x}}_{t-delay}$): the instrument variables. All
the rest of the predetermined variables are fixed:\
$\bar{\textbf{y}}_t= \textbf{G}(\bar{\textbf{x}}_{t-delay})$

If we invert G we have a model where instruments are functions of
targets:
$\bar{\textbf{x}_{t-delay}}= \textbf{G}^{-1}(\bar{\textbf{y}_{t}})$.
Then all we have to do is to find
$\textbf{G}^{-1}(\bar{\textbf{y}_{t}})$

The approximated Jacobi matrix of $\textbf{G}$ :
$\textbf{J}_t \approx \frac{\Delta \textbf{G} }{\Delta \bar{\textbf{x}}_{t-delay}}$
is used to find the instruments
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
### And how to solve for the instruments

For most models
$\bar{\textbf{x}}_{t-delay}= \textbf{G}^{-1}(\bar{\textbf{y}_{t}})$ do
not have a nice close form solution. However it can be solved
numerically. We turn to Newton--Raphson method.

So $\bar{\textbf{x}}_{t-delay}= \textbf{G}^{-1}(\bar{\textbf{y}_{t}^*})$
will be found using :

for $k$ = 1 to convergence

> $\bar{\textbf{x}}_{t-delay,end}^k= \bar{\textbf{x}}_{t-delay,end}^{k-1}+ \textbf{J}^{-1}_t \times (\bar{\textbf{y}_{t}^*}- \bar{\textbf{y}_{t}}^{k-1})$

> $\bar{\textbf{y}}_t^{k}= \textbf{G}(\bar{\textbf{x}}_{t-delay}^{k})$

convergence:
$\mid\bar{\textbf{y}_{t}^*}- \bar{\textbf{y}_{t}} \mid\leq \epsilon$

Now we just need to find:

$\textbf{J}_t = \frac{\partial \textbf{G} }{\partial \bar{\textbf{x}}_{t-delay}}$

A number of differentiation methods can be used (symbolic, automated or
numerical). ModelFlow uses numerical differentiation, as it is quite
simple and fast.

$\textbf{J}_t \approx \frac{\Delta \textbf{G} }{\Delta \bar{\textbf{x}}_{t-delay}}$

That means that we should run the model one time for each instrument,
and record the effect on each of the targets, then we have
$\textbf{J}_t$

In order for $\textbf{J}_t$ to be invertible there has to be the same
number of targets and instruments.

However, each instrument can be a basket of exogenous variable. They
will be adjusted in fixed proportions. This can be useful for instance
when using bank leverage as instruments. Then the leverage instrument
can consist of several loan types.

You will notice that the level of $\bar{\textbf{x}}$ is updated (by
$\textbf{J}^{-1}_t \times (\bar{\textbf{y}_{t}^*}- \bar{\textbf{y}_{t}}^{k-1})$)
in all periods from $t-delay$ to $end$, where $end$ is the last
timeframe in the dataframe. This is useful for many applications
including calibration of disease spreading models and in economic
models, where the instruments are level variable (i.e. not change
variables). If this is not suitable, it can be changed in a future
release.

The target/instrument functionality is implemented in the python class
`targets_instruments` specified in **ModelFlows** `modelinvert` module.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
### An example

The workflow is as follow:

1.  Define the targets
2.  Define the instruments
3.  Create a target_instrument class istance
4.  Solve the problem

Step one is to define the targets. This is done by creating a dataframe
where the target values are set.
:::

::: {.cell .code execution_count="36" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
msolow.basedf
```

::: {.output .execute_result execution_count="36"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>L</th>
      <th>K</th>
      <th>ALFA</th>
      <th>A</th>
      <th>DEPRECIATES_RATE</th>
      <th>LABOR_GROWTH</th>
      <th>SAVING_RATIO</th>
      <th>I</th>
      <th>K_I</th>
      <th>Y</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>101.000000</td>
      <td>100.025580</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>5.025580</td>
      <td>0.990352</td>
      <td>100.511609</td>
      <td>95.486029</td>
    </tr>
    <tr>
      <th>2</th>
      <td>102.010000</td>
      <td>100.076226</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>5.051924</td>
      <td>0.981043</td>
      <td>101.038487</td>
      <td>95.986562</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103.030100</td>
      <td>100.151443</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>5.079029</td>
      <td>0.972060</td>
      <td>101.580575</td>
      <td>96.501546</td>
    </tr>
    <tr>
      <th>4</th>
      <td>104.060401</td>
      <td>100.250762</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>5.106891</td>
      <td>0.963390</td>
      <td>102.137821</td>
      <td>97.030930</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>257.353755</td>
      <td>185.913822</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>10.936821</td>
      <td>0.722406</td>
      <td>218.736417</td>
      <td>207.799596</td>
    </tr>
    <tr>
      <th>96</th>
      <td>259.927293</td>
      <td>187.661027</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>11.042896</td>
      <td>0.721975</td>
      <td>220.857924</td>
      <td>209.815028</td>
    </tr>
    <tr>
      <th>97</th>
      <td>262.526565</td>
      <td>189.428077</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>11.150101</td>
      <td>0.721558</td>
      <td>223.002024</td>
      <td>211.851922</td>
    </tr>
    <tr>
      <th>98</th>
      <td>265.151831</td>
      <td>191.215119</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>11.258446</td>
      <td>0.721153</td>
      <td>225.168912</td>
      <td>213.910466</td>
    </tr>
    <tr>
      <th>99</th>
      <td>267.803349</td>
      <td>193.022302</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>11.367939</td>
      <td>0.720761</td>
      <td>227.358789</td>
      <td>215.990850</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 11 columns</p>
</div>
```
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
**Define Targets**
:::

::: {.cell .code execution_count="37" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
target = msolow.basedf.loc[50:,['L','K']]+[30,10]
target.head()
```

::: {.output .execute_result execution_count="37"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>L</th>
      <th>K</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>194.463182</td>
      <td>135.971544</td>
    </tr>
    <tr>
      <th>51</th>
      <td>196.107814</td>
      <td>136.933236</td>
    </tr>
    <tr>
      <th>52</th>
      <td>197.768892</td>
      <td>137.911105</td>
    </tr>
    <tr>
      <th>53</th>
      <td>199.446581</td>
      <td>138.905161</td>
    </tr>
    <tr>
      <th>54</th>
      <td>201.141047</td>
      <td>139.915414</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
Then we have to provide the instruments. This is **a list of list of
tuples**.

-   Each element in the outer list is an instrument.
-   Each element in the inner list is an instrument variable
-   Each element of the tuple contains a variable name and the
    associated impulse $\Delta$.

The $\Delta variable$ is used in the numerical differentiation. Also if
one instrument contains several variables, the proportion of each
variable will be determined by the relative $\Delta variable$.

For this experiment the inner list only contains one variable.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
**Define Instruments**
:::

::: {.cell .code execution_count="38" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
instruments = [ [('LABOR_GROWTH',0.001)] , [('DEPRECIATES_RATE',0.001)]]
```
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
**Run the experiment**

For models which are relative linear we don\'t need to update 𝐉𝑡 for
each iteration and time frame. As our small toy model is nonlinear, the
jacobi matrix has to be updated frequently. This is controlled by the
nonlin=True option below.
:::

::: {.cell .code execution_count="39" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
result = msolow.invert(msolow.lastdf,target,instruments,nonlin=True)
```

::: {.output .display_data}
``` {.json}
{"model_id":"b46300aea3a040d2b5d4bfe43704cbd5","version_major":2,"version_minor":0}
```
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
And do the result match the target?
:::

::: {.cell .code execution_count="40" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
(result-target).loc[50:,['L','K']].plot();
```

::: {.output .display_data}
![](ac454ad188754bde5891a0b80bc0219fd2d9a433.png)
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
So we got results for the target variable very close to the target
values.
:::

::: {.cell .code execution_count="41" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
msolow.smpl(90,100)  # we only want a few years 

msolow.basedf.Y
```

::: {.output .execute_result execution_count="41"}
    0       0.000000
    1     100.511609
    2     101.038487
    3     101.580575
    4     102.137821
             ...    
    95    218.736417
    96    220.857924
    97    223.002024
    98    225.168912
    99    227.358789
    Name: Y, Length: 100, dtype: float64
:::
:::

::: {.cell .code execution_count="42" slideshow="{\"slide_type\":\"skip\"}"}
``` {.python}
msolow.lastdf.Y
```

::: {.output .execute_result execution_count="42"}
    0       0.000000
    1     100.511609
    2     101.038487
    3     101.580575
    4     102.137821
             ...    
    95    237.269232
    96    239.389730
    97    241.532869
    98    243.698846
    99    245.887858
    Name: Y, Length: 100, dtype: float64
:::
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
### Shortfall targets

Above the target for each target variable is a certain values. Sometime
we we need targets being above a certain shortfall value. In this case
an instrument should be used to make the achieve the target threshold
only if the target is belove the target. This is activated by an
option:**shortfall=True**.

This feature can be useful calculating the amount of deleverage needed
for banks to achieve a certain threshold of capital.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
## Attribution / Explanation {#attribution--explanation}

Experience shows that it is useful to be able to explain the difference
between the result from two runs. The first level of understanding the
difference is to look at selected formulas and find out, how much each
input variables accounts for. The second level of understanding the
difference is to look at the attribution of the exogenous variables to
the results of the model.

If we have:

$y = f(a,b)$

and we have two solutions where the variables differs by
$\Delta y, \Delta a, \Delta b$

How much of $\Delta y$ can be explained by $\Delta a$ and $\Delta b$ ?

Analytical the attributions $\Omega a$ and $\Omega b$ can be calculated
like this:

$\Delta y = \underbrace{\Delta a \dfrac{\partial {f}}{\partial{a}}(a,b)}_{\Omega a} + \underbrace{\Delta b \dfrac{\partial {f}}{\partial{b}}(a,b)}_{\Omega b}+Residual$
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
ModelFlow will do a numerical approximation of $\Omega a$ and
$\Omega b$. This is done by looking at the two runs of the model:

\\begin{eqnarray}\
y_0&=&f(a\_{0},b\_{0}) \\ y_1&=&f(a_0+\\Delta a,b\_{0}+ \\Delta b)
\\end{eqnarray}

So $\Omega a$ and $\Omega b$ can be determined:

\\begin{eqnarray}\
\\Omega f_a&=&f(a_1,b_1 )-f(a_1-\\Delta a,b_1) \\ \\Omega
f_b&=&f(a_1,b_1 )-f(a_1,b_1-\\Delta b) \\end{eqnarray}

And:

\\begin{eqnarray} residual = \\Omega f_a + \\Omega f_b -(y_1 - y_0)
\\end{eqnarray} If the model is fairly linear, the residual will be
small.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
### Formula attribution

Attribution analysis on the formula level is performed by the method
**.dekomp**.

This method utilizes that two attributes .basedf and .lastdf containing
the first and the last run are contained in the model instance. Also all
the formulas are contained in the instance. Therefore a model, just with
one formula - is created. Then experiments mentioned above is run for
each period and each right hand side variable.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
### Model Attribution

At the model level we start by finding which exogenous variables have
changed between two runs.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
## Python functions can be incorporated

### A mean variance problem

If we look at a fairly general mean variance optimization problem which
has been adopted to banks it looks like this:

\\begin{eqnarray}\
\\mathbf x & &\\mbox{Position in each asset(+)/liability(-) type}\\
\\mathbf x & &\\mbox{Position in each asset(+)/liability(-) type}\\
\\mathbf \\Sigma & &\\mbox{Covariance matrix} \\ \\mathbf r &
&\\mbox{Return vector}\\ \\lambda & &\\mbox{Risk aversion}\\
\\mathbf{riskweights}& &\\text{Vector of risk weights, liabilities has
riskweight = 0}\\ Capital& &\\mbox{Max of sum of risk weighted assets}\\
\\mathbf{lcrweights}& &\\text{Vector of LCR weights, liabilities has
lcrweight = 0}\\ LCR& &\\text{Min of sum of lcr weighted assets}\\
\\mathbf{leverageweight}&&\\text{Vector of leverage weights, liabilities
has leverageweight = 0}\\ Equity&&\\mbox{Max sum of leverage weighted
positions}\\ Budget&&\\mbox{initial sum of the positions}\\
\\end{eqnarray}

\\begin{eqnarray} \\mbox{minimize:} & \\lambda \\mathbf x\^T \\mathbf
\\Sigma \\mathbf x - (1-\\lambda) \\mathbf r\^T \\mathbf x & \\mbox{If
}\\lambda \\mbox{ = 1 minimize risk, if } \\lambda\\mbox{ = 0 maximize
return }\\ \\mbox{subject to:} & \\mathbf x \\succeq \\mathbf{x\^{min}}
&\\mbox{Minimum positions}\\ & \\mathbf x \\preceq \\mathbf{x\^{max}}
&\\mbox{Maximum positions}\\ & \\mathbf{riskweights}\^T\\mathbf x \\leq
Capital &\\mbox{Risk weighted assets \<= capital}\\ &
\\mathbf{lcrweights}\^T\\mathbf x \\geq LCR &\\mbox{lcr weighted assets
\>= LCR target}\\ & \\mathbf{leverageweight}\^T\\mathbf x \\leq equity
&\\mbox{leverage weighted assets \<= equity}\\ & \\mathbf 1\^T\\mathbf x
= Budget & \\mbox{Sum of positions = B} \\end{eqnarray}
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
### The mean variance problem in the business language language

Wrap optimizing in the CVX library into a Python function:
`<br>`{=html}In the business logic language this problem can be
specified like this:

    positions =  mv_opt(msigma,return,riskaversion, budget, 
                [[risk_weights] , [-lcr_weights] , [leverage_weights]],
                  [capital, -lcr , equity] ,min_position,max_position) 

Where the arguments are appropriately dimensioned CVX matrices and
vectors.

For a more elaborate example there is an special notebook on the subject
of optimization.

Also it should be mentioned that there is an expansion of the basic
problem taking transaction cost into account.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
## Stability

Jacobi matrices can be used to evaluate the stability properties of the
model. To do this we first look at a linearized version of the model. We
are interested in the effect of shocks to the system. Will shocks be
damped or will they be amplified.
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
## Live models

**Showtime**
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"slide\"}"}
# Summary

ModelFlow allows easy implementation of models in Python, Which is a
powerful and agile language. ModelFlow leverage on the rich ecosystem of
Python in order to:

-   Separates the specification of a model and the code which solves the
    model. So the user can concentrate on the economic and not the
    implementation of the model.
-   Can include user specified Python function in the model definition.
-   Can solve very large m,odels
-   Can solve simultaneous models.
-   Keeps tab on the dependencies of the formulas. This allows for easy
    Tracing of results.
-   Can perform model inversion (goal seek) with multiple targets and
    instruments
-   Can attribute changes in results to input variables. Both for
    individual formulas and the complete model
-   Can include optimizing behavior

The purpose of this notebook has been to give a broad introduction to
model management using ModelFlow. Using the tool requires some knowledge
of python. The required knowledge depends on the complexity of the
model. So ModelFlow can be used in Python training.

To get more in-depth knowledge there is a Sphinx based documentation of
the library. There you can find the calling conventions and
documentation of all elements.

All suggestions and recommendations are welcome
:::

::: {.cell .markdown slideshow="{\"slide_type\":\"skip\"}"}
# Literature:

Aho, Lam, Sethi, Ullman (2006), Compilers: Principles, Techniques, and
Tools (2nd Edition), Addison-Wesley

Berndsen, Ron (1995), [Causal ordering in economic
models](https://kundoc.com/pdf-causal-ordering-in-economic-models-.html),
Decision Support Systems 15 (1995) 157-165

Danmarks Nationalbank (2004), [MONA \-- a quarterly model of Danish
economy](http://www.nationalbanken.dk/da/publikationer/Documents/2003/11/mona_web.pdf)

Denning, Peter J. (2006), [The Locality
Principle](http://denninginstitute.com/pjd/PUBS/locality_2006.pdf),
Chapter in *Communication Networks and Systems* (J Barria, Ed.).
Imperial College Press

Gilli, Manfred (1992), [Causal Ordering and Beyond, International
Economic Review, Vol. 33, No. 4 (Nov., 1992), pp.
957-971](http://www.jstor.org/stable/2527152?seq=1#page_scan_tab_contents)

McKinney, Wes (2011),\[pandas: a Foundational Python Library for Data
Analysis and Statistics,\] Presented at
PyHPC2011\](<http://www.scribd.com/doc/71048089/pandas-a-Foundational-Python-Library-for-Data-Analysis-and-Statistics>)

Numba (2015) documentation,
[http://numba.pydata.org/numba-doc/0.20.0/user/index.html](http://numba.pydata.org/numba-doc/0.20.0/user/index.html%20)

Pauletto, G. (1997), [Computational Solution of Large-Scale
Macroeconometric
Models](http://link.springer.com/book/10.1007%2F978-1-4757-2631-2), ISBN
9781441947789

E. Petersen, Christian & A. Sims, Christopher. (1987). Computer
Simulation of Large-Scale Econometric Models: Project Link.
International Journal of High Performance Computing Applications
(<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.853.6387&rep=rep1&type=pdf>)

Tinberger, Jan (1956), Economic policy: Principles and Design,
Amsterdam,

# Footnotes

$^1$: The author has been able to draw on experience creating software
for solving the macroeconomic models ADAM in Hansen Econometric.

$^2$: In this work a number of staff in Danmarks Nationalbank made
significant contributions: Jens Boldt and Jacob Ejsing to the program.
Rasmus Tommerup and Lindis Oma by being the first to implement a stress
test model in the system.

$^3$: In ECB Marco Gross, Mathias Sydow and many other collegues has
been of great help.

$^4$: The system has benefited from discussions with participants in
meetings at: IMF, Bank of Japan, Bank of England, FED Board, Oxford
University, Banque de France, Single Resolution Board

$^5$: Ast stands for: [Abstract Syntax
Tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree)

$^6$: Re stands for[: Regular
expression](https://en.wikipedia.org/wiki/Regular_expression)
:::

::: {.cell .code}
``` {.python}
```
:::
