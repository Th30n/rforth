\ Put execution token (xt) on stack; parses at run-time
\ For parsing during compile-time use word [']
: ' WORD FIND >CFA ;

\ Put execution token (xt) on stack; parses at compile-time
\ For parsing during run-time use word '
: ['] IMMEDIATE
    LIT LIT ,
;

\ Compile a `word` that would otherwise be IMMEDIATE.
: [COMPILE] IMMEDIATE
    WORD FIND >CFA ,
;

\ Take whatever `val` is on stack and compile LIT `val`.
: LITERAL IMMEDIATE
    ['] LIT , \ compile LIT
    ,         \ compile `val` from stack
;

: CELL+ 1 CELLS + ;

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
  C@    \ WORD should invoke BYE on zero length string, so this is safe
;

\ CHAR for parsing at compile-time
: [CHAR] IMMEDIATE  CHAR [COMPILE] LITERAL ;

\ Put blank (space) on stack.
: BL  32 ;

: SPACE BL EMIT ;

\ Emit carriage return (\n here).
: CR  10 EMIT ;

\ CONTROL STRUCTURES
\ ==================
\ These only work inside compiled words. Typing in immediate mode (as presented
\ by the interpreter) will result in an error.

: IF IMMEDIATE
    ['] 0BRANCH , \ compile 0BRANCH
    HERE \ @    \ save location of the offset on stack
    0 ,         \ compile dummy offset
;

: THEN IMMEDIATE
    DUP
    HERE SWAP - \ calculate the offset from the address save on stack
    SWAP !      \ store the offset in the back-filled location
;

: ELSE IMMEDIATE
    ['] BRANCH ,  \ definite branch to just over the false part
    HERE        \ save location of the offset
    0 ,         \ compile dummy offset
    SWAP        \ now back-fill the original (IF) offset
    DUP         \ same as for THEN above
    HERE SWAP -
    SWAP !
;
