: TRUE  -1 ;
: FALSE 0 ;

\ Put execution token (xt) on stack; parses at compile-time
\ For parsing during run-time use word '
: ['] IMMEDIATE
  ' LIT ,
;

\ Overwrite original ' to now parse at run-time.
: ' WORD FIND >CFA ;

\ Compile a `word` that would otherwise be IMMEDIATE.
: [COMPILE] IMMEDIATE
    WORD FIND >CFA ,
;

\ Take whatever `val` is on stack and compile LIT `val`.
: LITERAL IMMEDIATE
    ['] LIT , \ compile LIT
    ,         \ compile `val` from stack
;

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
