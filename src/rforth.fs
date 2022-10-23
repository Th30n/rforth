: TRUE  -1 ;
: FALSE 0 ;

\ Put execution token (xt) on stack; parses at compile-time
\ For parsing during run-time use word '
: ['] IMMEDIATE
  ' LIT ,
;

\ Overwrite original ' to now parse at run-time.
: ' WORD FIND >CFA ;

\ Take whatever `val` is on stack and compile LIT `val`.
: LITERAL IMMEDIATE
    ['] LIT , \ compile LIT
    ,         \ compile `val` from stack
;

\ Compile a `word` that would otherwise be IMMEDIATE.
: [COMPILE] IMMEDIATE
    WORD FIND >CFA ,
;

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
