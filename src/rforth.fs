\ Put execution token (xt) on stack; parses at run-time
\ For parsing during compile-time use word [']
: '  WORD FIND >CFA ;

\ Put execution token (xt) on stack; parses at compile-time
\ For parsing during run-time use word '
: ['] IMMEDIATE
    LIT LIT ,
;

\ Compile a `word` that would otherwise be IMMEDIATE.
: POSTPONE IMMEDIATE
    WORD FIND >CFA ,
;

\ Take whatever `val` is on stack and compile LIT `val`.
: LITERAL IMMEDIATE
    ['] LIT , \ compile LIT
    ,         \ compile `val` from stack
;

: CELL+  1 CELLS + ;

\ Compile a recursive call to word currently being defined.
: RECURSE IMMEDIATE  LATEST >CFA , ;

\ DOES> will modify the latest CREATEd word.
\ It can be only used in a colon definition.
: DOES> IMMEDIATE
    ['] LIT ,     \ compile LIT
    HERE          \ save location where we will store ptr to words after DOES>
    0 ,           \ store dummy 0 as ptr
    ['] LATEST ,  \ compile LATEST
    ['] >CFA ,    \ compile >CFA
    ['] CELL+ ,   \ compile skipping over the codeword to ptr for `docreate`
    ['] ! ,       \ compile overwriting ptr with HERE
    ['] EXIT ,    \ compile EXIT to skip out of executing words after DOES>
    HERE SWAP !   \ backfill the ptr with location of words after DOES>
;

\ Get data field address from an execution token
: >BODY  2 CELLS + ;

: VARIABLE  CREATE 0 , ;

: CONSTANT
    CREATE ,
DOES>
    @
;

-1 CONSTANT TRUE
0 CONSTANT FALSE

\ Read a word and put the first character on stack.
: CHAR
  WORD
  DROP  \ Drop the length part
  C@    \ If read len was 9, this will read some memory after the input
;

\ CHAR for parsing at compile-time
: [CHAR] IMMEDIATE  CHAR POSTPONE LITERAL ;

\ Put blank (space) on stack.
: BL  32 ;

: SPACE  BL EMIT ;

\ Emit carriage return (\n here).
: CR  10 EMIT ;

\ CONTROL STRUCTURES
\ ==================
\ These only work inside compiled words. Typing in immediate mode (as presented
\ by the interpreter) will result in an error.

: IF IMMEDIATE
    ['] 0BRANCH , \ compile 0BRANCH
    HERE          \ save location of the offset on stack
    0 ,           \ compile dummy offset
;

: THEN IMMEDIATE
    DUP
    HERE SWAP - \ calculate the offset from the address save on stack
    SWAP !      \ store the offset in the back-filled location
;

: ELSE IMMEDIATE
    ['] BRANCH ,  \ definite branch to just over the false part
    HERE          \ save location of the offset
    0 ,           \ compile dummy offset
    SWAP          \ now back-fill the original (IF) offset
    DUP           \ same as for THEN above
    HERE SWAP -
    SWAP !
;

: BEGIN IMMEDIATE  HERE ;

: UNTIL IMMEDIATE
    ['] 0BRANCH , \ if test was false, jump to BEGIN
    HERE - ,      \ offset is BEGIN's location minus HERE.
;

\ Infinite loop for BEGIN
: AGAIN IMMEDIATE
    ['] BRANCH ,
    HERE - ,
;

: WHILE IMMEDIATE
    ['] 0BRANCH , \ if test was false, jump to location set by REPEAT
    HERE          \ save location of the offset
    0 ,           \ compile dummy offset
;

: REPEAT IMMEDIATE
    ['] BRANCH ,  \ jump to original BEGIN's location
    SWAP HERE - , \ offset to BEGIN's location
    DUP
    HERE SWAP -   \ offset to be filled for WHILE
    SWAP !
;

\ END CONTROL STRUCTURES
\ ======================

\ TODO: Make these builtin?
\ ( x1 x2 -- x1 x2 x1 )
: OVER  >R DUP R> SWAP ;
\ ( x1 x2 -- x2 x1 x2 )
: TUCK  SWAP OVER ;
: 2DROP  DROP DROP ;
\ ( x1 x2 -- x1 x2 x1 x2 )
: 2DUP  OVER OVER ;

: 1-  1 - ;
: 1+  1 + ;
: <>  = INVERT ;
: <  2DUP <> >R SWAP > R> AND ;
: 0=  0 = ;
: 0>  0 > ;
: 0<  0 < ;
: 0<> 0 <> ;

\ Add n|u to single cell number at a-addr
\ ( n|u a-addr -- )
: +!
    DUP @ \ ( n|u a-addr orig )
    ROT + \ ( a-addr new )
    SWAP !
;

\ Get total source length and current address of input
\ ( u c-addr )
: CURR-SOURCE
  SOURCE \ ( input-addr input-len )
  SWAP >IN @ +
;

\ Calculate len of c-addr to current >IN
\ ( c-addr -- u )
: IN-LEN-FROM
    CURR-SOURCE NIP \ ( c-addr end-addr )
    SWAP -
;

\ Parse `ccc` delimited by `char`
\ ( char "ccc<char" -- c-addr u )
: PARSE
    CURR-SOURCE     \ ( char input-len c-addr )
    >R              \ store c-addr on return stack for later
    BEGIN           \ ( char input-len ) ( R: c-addr )
      >IN @ U>      \ input-len > input-ix
    WHILE           \ ( char ) ( R: c-addr )
      CURR-SOURCE   \ ( char input-len curr-addr ) ( R: c-addr )
      C@ ROT        \ ( input-len curr-char char ) ( R: c-addr )
      1 >IN +!      \ increment >IN
      TUCK          \ ( input-len char curr-char char ) ( R: c-addr )
      = IF          \ ( input-len char ) ( R: c-addr )
        2DROP
        R> DUP      \ ( c-addr c-addr )
        IN-LEN-FROM \ ( c-addr u )
        EXIT
      THEN          \ ( input-len char ) ( R: c-addr )
      SWAP
    REPEAT          \ ( char input-len ) ( R: c-addr )
    2DROP
    R> DUP
    IN-LEN-FROM
;

: ( IMMEDIATE  [CHAR] ) PARSE 2DROP ;

( We can now write comments with ( ... )
( NOTE: Nesting is not supported       )
( ==================================== )

\ Display a character string.
( c-addr u -- )
: TYPE
    OVER + SWAP
    BEGIN     ( end-addr c-addr )
      2DUP U>
    WHILE     ( end-addr c-addr )
      DUP C@  ( end-addr c-addr char )
      EMIT    ( end-addr c-addr )
      1+      \ increment c-addr
    REPEAT    ( end-addr c-addr )
    2DROP
;
