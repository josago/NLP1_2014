#!/usr/bin/perl

use strict;
use warnings;

use LWP::Simple;
use Lingua::Stem::Any; # Install executing within CPAN: "install Lingua::Stem::Any"

my %DATABASE; # Final database with all the information.

sub crawlIMDb
{
    my %attributes;

    my $name = shift @_;

    my $code = get("http://www.imdb.com/find?q=$name&&s=tt&&ttype=ft&ref_=fn_ft");

    $code =~ /class=\"result_text\"> <a href=\"\/title\/tt(\d+)/s;
    
    my $number = $1;

    $code = get("http://www.imdb.com/title/tt$number");
    
#     $code =~ /<span class=\"itemprop\" itemprop=\"name\">(.+?)<\/span>/s;
# 
#     $attributes{'NAME'} = $1;
    
#     $code =~ /<a href=\"\/year\/(\d{4})\/\?ref_=tt_ov_inf\"/s;
#     
#     $attributes{'YEAR'} = $1;
    
    if ($code =~ /<span itemprop=\"ratingValue\">([\d\.]+?)<\/span>/s)
    {
        $attributes{'SCORE'} = $1;
    }
    else
    {
        return undef;
    }
    
#     if ($code =~ /<span itemprop=\"contentRating\">(.+?)<\/span>/s)
#     {
#         $attributes{'RATING'} = $1;
#     }
    
#     $code =~ /itemprop=\"director\".+?itemprop=\"name\">(.+?)<\/span>/s;
#     
#     $attributes{'DIRECTOR'} = $1;
    
    $code =~ /<div class="inline canwrap" itemprop="description">.*?<p>(.+?)<em class="nobr">/s;
    
    $attributes{'SUMMARY'} = $1;
    
#     $code =~ /<h4 class=\"inline\">Stars\:<\/h4>.*?itemprop=\"name\">(.+?)<\/span><\/a>(.*?itemprop=\"name\">(.+?)<\/span><\/a>)?(.*?itemprop=\"name\">(.+?)<\/span><\/a>)?.*?<span class=\"ghost\">/s;
#     
#     unless ($5)
#     {
#         unless ($3)
#         {
#             $attributes{'STARS'} = $1;
#         }
#         else
#         {
#             $attributes{'STARS'} = "$1,$3";
#         }
#     }
#     else
#     {
#         $attributes{'STARS'} = "$1,$3,$5";
#     }
    
    return %attributes;
}

sub populateDB
{
    opendir(my $scripts, 'scripts/') or die "Could not open directory 'scripts/': $!\n";

    while (readdir($scripts))
    {
        my $filename = $_;
        
        if ($filename =~ /(.+?)\.html/) # Avoids "dot" files ./ and ../
        {
            my $name = $1;
            
            print "Parsing movie '$name'...\n";
            
            $DATABASE{$name} = {};
            
            print "\tReading movie script...\n";
            
            open(my $file, '<', "scripts/$filename") or die "Could not open file 'scripts/$filename': $!\n";
            
            my $script = '';
            
            while (<$file>)
            {
                $script .= $_;
            }
            
            close($file);
        
            $DATABASE{$name}{'SCRIPT'} = $script;
        
            print "\tCrawling IMDb for more information about the movie...\n";
            
            my %attributes = crawlIMDb($name);
            
            if (%attributes ~~ undef)
            {
                delete $DATABASE{$name};
            }
            else
            {
                foreach (keys %attributes)
                {
                    $DATABASE{$name}{$_} = $attributes{$_};
                }
            }
        }
    }

    close($scripts);
}

sub makeScriptFeatures
{
    my %words_global; # Number of occurrences of each word.
    my %words_films;  # Number of times each word appears at least once in a movie.
    
    my $stemmer = Lingua::Stem::Any->new(
        language => 'en',
        source   => 'Lingua::Stem::Snowball',
    );

    print "Making movie language features...\n";

    foreach my $name (keys %DATABASE)
    {
        my @stuff = ('FEATURES_SCRIPT', 'SCRIPT', 'FEATURES_SUMMARY', 'SUMMARY');
        
        while (@stuff)
        {
            my $out = shift @stuff;
            my $in  = shift @stuff;
        
            $DATABASE{$name}{$out} = {};
        
            $DATABASE{$name}{$in} =~ s/<[^<>]+?>//g; # Get rid of HTML tags.
            
            my @script_tokens = split(/[\s,;\.:\-!\(\)\"'\?\/_\[\]\*]+/s, $DATABASE{$name}{$in});
            
            foreach (@script_tokens)
            {
                my $token = lc($_); # Token to lowercase.
                
                $token = $stemmer->stem($token); # Use stemmer.

                $DATABASE{$name}{$out}{$token}++;
            }
            
            foreach my $word (keys %{$DATABASE{$name}{$out}})
            {
                $words_global{$word} += $DATABASE{$name}{$out}{$word};
                
                $words_films{$word}++;
            }
        }
    }
    
    print "No stop-list:\n";
    
    my $sum = 0;
    
    foreach (keys %words_global)
    {
        $sum += $words_global{$_};
    }

    print "\tTotal words = $sum\n";
    print "\tTotal unique words = " . scalar(keys %words_global) . "\n";
    
    print "With stop-list:\n";
    
    my @stoplist = ('a', 'o', 'm', 's', 't', 'v', 'an', 'as', 'and', 'by', 'or', 'but', 'the', 'in', 'of', 'at', 'to', 'on', 'for', 'it', 'that', 'is', 'are', 're', 'with');

    foreach my $word (@stoplist)
    {
        $word = $stemmer->stem($word); # Use stemmer also on the stop-words.
    
        $sum -= $words_global{$word};
    
        delete $words_global{$word};
        delete $words_films{$word};
        
        foreach my $name (keys %DATABASE)
        {
            delete $DATABASE{$name}{'FEATURES_SUMMARY'}{$word};
            delete $DATABASE{$name}{'FEATURES_SCRIPT'}{$word};
        }
    }
    
    print "\tTotal words = $sum\n";
    print "\tTotal unique words = " . scalar(keys %words_global) . "\n";

    print "With prunning:\n";
    
    foreach my $word (keys %words_global)
    {
        if ($words_films{$word} <= 1 or $words_global{$word} <= 1 or $word =~ /^\d+$/) # Word prunning rules.
        {
            $sum -= $words_global{$word};
        
            delete $words_global{$word};
            delete $words_films{$word};
        
            foreach my $name (keys %DATABASE)
            {
                delete $DATABASE{$name}{'FEATURES_SUMMARY'}{$word};
                delete $DATABASE{$name}{'FEATURES_SCRIPT'}{$word};
            }
        }
    }
    
    print "\tTotal words = $sum\n";
    print "\tTotal unique words = " . scalar(keys %words_global) . "\n";

    return (\%words_global, \%words_films);
}

sub writeDB
{
    my ($words_global, $words_films) = @_;

    open(my $tokens, '>', 'tokens.txt') or die "Could not write into file 'tokens.txt': $!\n";
    
    my @list_tokens = keys %{$words_global};
    
    foreach my $word (@list_tokens)
    {
        print $tokens "$word\t${$words_global}{$word}\t${$words_films}{$word}\n";
    }
    
    close($tokens);
    
    open(my $database, '>', 'database.csv') or die "Could not write into file 'database.csv': $!\n";
    
    foreach my $name (keys %DATABASE)
    {
        my $features_summary = '';
        my $features_script  = '';

        foreach my $word (@list_tokens)
        {
            if (exists $DATABASE{$name}{'FEATURES_SUMMARY'}{$word})
            {
                $features_summary .= "$DATABASE{$name}{'FEATURES_SUMMARY'}{$word},";
            }
            else
            {
                $features_summary .= '0,';
            }
        
            if (exists $DATABASE{$name}{'FEATURES_SCRIPT'}{$word})
            {
                $features_script .= "$DATABASE{$name}{'FEATURES_SCRIPT'}{$word},";
            }
            else
            {
                $features_script .= '0,';
            }
        }
        
        $features_script =~ s/,$//; # Remove trailing comma.

        print $database "$name\t$DATABASE{$name}{'SCORE'}\t$features_summary\t$features_script\n"; # Most basic features.
    }
    
    close($database);
}

# Main program:

populateDB();
my ($words_global, $words_films) = makeScriptFeatures();
writeDB($words_global, $words_films);