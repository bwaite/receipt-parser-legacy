Just a stripped down version of the parser here https://github.com/ReceiptManager/receipt-parser-legacy

Found that several dependencies were redundant when scipy and opencv were already in place
(And I should be able to remove the rotate used from scipy). Also just tried to simplify things
a bit. Will try and bring back my test cases later
