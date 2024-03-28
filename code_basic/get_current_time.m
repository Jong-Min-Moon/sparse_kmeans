function current_time = get_current_time()
    import java.util.TimeZone 
    nn = now;
    ds = datestr(nn);
    current_time = datetime(ds,'TimeZone',char(TimeZone.getDefault().getID()));
end