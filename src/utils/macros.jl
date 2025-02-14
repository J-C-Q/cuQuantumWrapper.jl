macro domain(args...)
    # Accept either one or two arguments:
    # 1. condition
    # 2. condition, message
    if length(args) == 1
        condition = args[1]
        msg = "Domain check failed: " * string(condition)
    elseif length(args) == 2
        condition = args[1]
        msg = args[2]
    else
        throw(ArgumentError("@domain macro accepts 1 or 2 arguments, got $(length(args))"))
    end

    # Convert the condition expression to a string (for reporting)
    cond_str = string(condition)

    return quote
        if !($(esc(condition)))
            throw(DomainError($cond_str, $(esc(msg))))
        end
    end
end

macro argument(args...)
    # Accept either one or two arguments:
    # 1. condition
    # 2. condition, message
    if length(args) == 1
        condition = args[1]
        msg = "Argument check failed: " * string(condition)
    elseif length(args) == 2
        condition = args[1]
        msg = args[2]
    else
        throw(ArgumentError("@argument macro accepts 1 or 2 arguments, got $(length(args))"))
    end

    # Convert the condition expression to a string (for reporting)
    cond_str = string(condition)

    return quote
        if !($(esc(condition)))
            throw(ArgumentError($cond_str * ": " * $(esc(msg))))
        end
    end
end
