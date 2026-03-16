# Modelflow Model Parsing, Analysis, and Dependency Graph Construction

This document explains how Modelflow converts `FRML` equations into the
internal representation used by the model solver. The process consists
of three main steps:

1.  **Tokenization** (`model_parse` in `modelpattern.py`)
2.  **Model analysis** (`analyzemodelnew` in `modelclass.py`)
3.  **Dependency graph and solution ordering**

------------------------------------------------------------------------

# Overview of the Parsing Pipeline

The pipeline from model text to internal model representation is:

    Model text
        ↓
    FRML extraction
        ↓
    model_parse()
        ↓
    token streams (nterm)
        ↓
    BaseModel.analyzemodelnew()
        ↓
    dependency graph
        ↓
    solver generation
        ↓
    model simulation

------------------------------------------------------------------------

# 1. `model_parse()` --- FRML Equation Tokenizer

Location:

    modelflow/modelpattern.py

## Purpose

`model_parse()` performs **lexical tokenization** of Modelflow `FRML`
equations.

It converts each equation into:

1.  formula metadata (`fatoms`)
2.  a flat list of tokens (`nterm`)

These tokens are later used by `analyzemodelnew()` to detect:

-   variable dependencies
-   lag/lead structure
-   left-hand side variables
-   equation structure

The tokenizer does **not build a syntax tree**.

------------------------------------------------------------------------

# Function Signature

``` python
model_parse(equations, funks=[])
```

### Parameters

  Parameter     Meaning
  ------------- ------------------------------------------------------
  `equations`   string containing one or more `FRML ... $` equations
  `funks`       optional list of extra function names

------------------------------------------------------------------------

# Return Value

`model_parse()` returns a list:

    [(fatoms, token_list), ...]

------------------------------------------------------------------------

# `fatoms` Structure

    namedtuple('fatoms', 'whole frml frmlname expression')

  Field          Meaning
  -------------- -------------------------------
  `whole`        full formula text
  `frml`         literal `"FRML"`
  `frmlname`     name and optional `<options>`
  `expression`   equation expression

Example:

    FRML GDP GDP = C + I + G $

------------------------------------------------------------------------

# Token Representation (`nterm`)

Tokens are stored as:

    nterm(number, op, var, lag)

Definition:

    namedtuple('nterm', ['number','op','var','lag'])

Only one of the first three fields is normally populated.

------------------------------------------------------------------------

# Token Fields

  Field      Meaning
  ---------- ---------------------------------
  `number`   numeric literal
  `op`       operator or function
  `var`      variable name
  `lag`      lag/lead attached to a variable

Empty fields are represented as `''`.

------------------------------------------------------------------------

# Supported Token Types

## Numbers

Examples:

    1
    3.14
    .5
    1e4
    1E-3

Example token:

    nterm('3.14','','','')

------------------------------------------------------------------------

## Operators

Stored in `op`.

Examples:

    =
    +
    -
    *
    /
    **
    (
    )
    ,
    [
    ]
    <
    >
    <=
    >=
    ==
    !=
    @
    |
    $
    .

Example:

    nterm('', '+', '', '')

------------------------------------------------------------------------

## Functions

Functions are treated as **operators** and stored in `op`.

Examples:

    LOG(
    EXP(
    ABS(
    MAX(
    MIN(
    DIFF(
    MOVAVG(

Functions from these modules are also recognized:

    modelBLfunk
    modeluserfunk
    modelclass.classfunk

Example:

    MAX(A,B)

produces:

    nterm('', 'MAX', '', '')

------------------------------------------------------------------------

## Variables

Variables appear in the `var` field.

Regex pattern:

    [A-Za-z_{][A-Za-z_{}0-9]*

Examples:

    GDP
    C
    YD
    BANK_LOANS
    FX_RATE

Token example:

    nterm('', '', 'GDP', '')

------------------------------------------------------------------------

## Lag / Lead Syntax

Lag and lead are attached to the **same token as the variable**.

Syntax:

    VAR(-n)
    VAR(+n)

Example:

    C(-1)
    GDP(+2)

Token:

    nterm('', '', 'C', '-1')

Special case:

    VAR(+0)
    VAR(-0)

are normalized to:

    lag=''

------------------------------------------------------------------------

# Example Tokenization

Input equation:

    FRML TEST A = B(-1) + 3.5 * MAX(C,D) $

Token stream:

    [
    nterm('', '', 'A', ''),
    nterm('', '=', '', ''),
    nterm('', '', 'B', '-1'),
    nterm('', '+', '', ''),
    nterm('3.5', '', '', ''),
    nterm('', '*', '', ''),
    nterm('', 'MAX', '', ''),
    nterm('', '(', '', ''),
    nterm('', '', 'C', ''),
    nterm('', ',', '', ''),
    nterm('', '', 'D', ''),
    nterm('', ')', '', '')
    ]

------------------------------------------------------------------------

# 2. `BaseModel.analyzemodelnew()`

Location:

    modelflow/modelclass.py

## Purpose

`analyzemodelnew()` converts the tokenized equations into the **internal
model structure** used by Modelflow.

It performs several tasks:

-   identify endogenous variables
-   extract variable dependencies
-   detect lags and leads
-   determine equation ordering
-   build metadata used by the solver

------------------------------------------------------------------------

# Inputs

`analyzemodelnew()` receives:

    fatoms
    token list (nterm objects)

from `model_parse()`.

------------------------------------------------------------------------

# Key Processing Steps

## Locate Assignment Operator

The assignment token is represented as:

    ('', '=', '', '')

Example code logic:

    assigpos = nt.index(self.aequalterm)

This splits the equation into:

    LHS  = tokens before "="
    RHS  = tokens after "="

------------------------------------------------------------------------

## Determine Endogenous Variable

The variable immediately before `=` is taken as the equation's
endogenous variable.

Example:

    GDP = C + I + G

endogenous variable:

    GDP

------------------------------------------------------------------------

## Extract Referenced Variables

Variables on the right-hand side are collected:

    [t.var for t in tokens if t.var]

These define **dependencies** of the equation.

------------------------------------------------------------------------

## Determine Lag and Lead Structure

Lag and lead values are extracted from:

    t.lag

Example:

    C(-1)
    GDP(+2)

The model stores:

    max_lag
    max_lead

for the entire model.

------------------------------------------------------------------------

## Store Equation Token Stream

The full list of tokens is stored internally for later use by:

-   solver generation
-   symbolic differentiation
-   attribution analysis

------------------------------------------------------------------------

# 3. Dependency Graph and Solution Ordering

After equations are analyzed, Modelflow constructs a **dependency
graph** between variables.

This graph determines:

-   equation evaluation order
-   simultaneous blocks
-   iterative solution requirements

------------------------------------------------------------------------

## Dependency Extraction

For each equation:

    LHS variable
    depends on
    RHS variables

Example:

    C = a*YD + b*C(-1)

Dependencies:

    C → YD
    C → C(-1)

Only **current-period variables** affect ordering.

Lagged variables do not create simultaneous dependencies.

------------------------------------------------------------------------

## Dependency Graph

Dependencies are represented as a directed graph:

    variable → variables it depends on

Example:

    GDP = C + I + G
    C = a*YD
    YD = W + TR

Graph:

    GDP → C
    GDP → I
    GDP → G
    C → YD
    YD → W
    YD → TR

------------------------------------------------------------------------

## Topological Sorting

If the graph contains **no cycles**, Modelflow can compute a solution
order using **topological sorting**.

Example order:

    W
    TR
    YD
    C
    GDP

This allows equations to be evaluated once per period.

------------------------------------------------------------------------

## Simultaneous Blocks

If the dependency graph contains cycles:

    A = B + 1
    B = A + 2

the variables form a **simultaneous block**.

These blocks require iterative solution methods.

------------------------------------------------------------------------

## Iterative Solution

Simultaneous systems are solved using repeated evaluation until
convergence.

Typical process:

    initial guess
        ↓
    evaluate equations
        ↓
    update variables
        ↓
    repeat until convergence

------------------------------------------------------------------------

# Example Complete Flow

Model:

    FRML CONS C = a*YD + b*C(-1) $
    FRML INC  YD = W + TR $
    FRML GDP  GDP = C + I + G $

Processing steps:

    FRML extraction
        ↓
    model_parse()
        ↓
    token streams
        ↓
    analyzemodelnew()
        ↓
    equation metadata
        ↓
    dependency graph
        ↓
    solution ordering
        ↓
    simulation engine

------------------------------------------------------------------------

# Developer Notes

When working with tokens:

    if t.var:
        variable reference

    if t.number:
        numeric constant

    if t.op:
        operator or function

    t.lag:
        lag/lead attached to variable

Understanding this representation is essential for modifying:

-   equation parsing
-   dependency analysis
-   solver generation
-   symbolic tools in Modelflow.
