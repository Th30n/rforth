fn main() {
    let mut forth = rforth::with_cli_args();
    rforth::run(&mut forth);
    dbg!(forth.data_stack());
    dbg!(forth.return_stack());
}
