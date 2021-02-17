def begins_with(s,t):
    return s[:len(t)]==t

def replace_extension(file_name,new_ext):
    parts = file_name.split('.')
    parts[-1]=new_ext
    return ".".join(parts)