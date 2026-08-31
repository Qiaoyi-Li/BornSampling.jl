import BornSampling
using Documenter

DocMeta.setdocmeta!(
    BornSampling,
    :DocTestSetup,
    quote
        import BornSampling
        const FiniteMPS = BornSampling.FiniteMPS
        const TK = BornSampling.TK
        const Random = BornSampling.Random
    end;
    recursive=true,
)

makedocs(
    modules=[BornSampling],
    authors="Qiaoyi Li",
    sitename="BornSampling.jl",
    remotes=nothing,
    checkdocs=:exports,
    format=Documenter.HTML(
        canonical="https://Qiaoyi-Li.github.io/BornSampling.jl",
        edit_link="main",
        repolink="https://github.com/Qiaoyi-Li/BornSampling.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Implementation" => "implementation.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo="github.com/Qiaoyi-Li/BornSampling.jl.git",
    devbranch="main",
)
