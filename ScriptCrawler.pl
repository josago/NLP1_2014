#!/usr/bin/perl

use strict;
use warnings;

use LWP::Simple;

my $code = get('http://www.imsdb.com/all%20scripts/');

while ($code =~ /<a href=\"\/Movie Scripts\/(.+?)\"/gi)
{
    print $1;
}

print $code;